import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from analysis.feature_engineering import add_indicators
from config import MODELS_DIR, LSTM_SEQ_LEN, XGB_N_ESTIMATORS, ML_LOOKAHEAD_BARS, ML_LABEL_THRESHOLD

MODELS_PATH = Path(MODELS_DIR)
MODELS_PATH.mkdir(exist_ok=True)

FEATURE_COLS = [
    "open", "high", "low", "close", "volume",
    "ema_8", "ema_21", "ema_50", "rsi", "atr",
    "macd", "macd_sig", "bb_upper", "bb_lower", "bb_mid",
    "stoch_k", "stoch_d",
    "body_ratio", "close_in_rng", "shadow_upper", "shadow_lower",
    "atr_ratio", "bb_width", "vol_ratio", "ret_1", "ret_3", "adx",
    "roc_5", "roc_10", "roc_20",
    "ema_slope_8", "ema_slope_21",
    "dist_ema_8", "dist_ema_21", "dist_ema_50",
    "cci", "wr", "mfi",
    "hist_vol_10", "hist_vol_20", "atr_pct",
    "inside_bar", "outside_bar", "bull_2", "bear_2",
    "dist_hh_10", "dist_ll_10", "donchian_pos",
    "autocorr_1", "skew_10",
    "hour_sin", "hour_cos", "dow",
]


def build_dataset(df: pd.DataFrame, lookahead: int = None) -> tuple:
    if lookahead is None:
        lookahead = ML_LOOKAHEAD_BARS
    df = df.copy()
    df = add_indicators(df)
    df["future_close"] = df["close"].shift(-lookahead)
    df["label"] = np.where(
        df["future_close"] > df["close"] * (1 + ML_LABEL_THRESHOLD), 1,
        np.where(df["future_close"] < df["close"] * (1 - ML_LABEL_THRESHOLD), 0, np.nan)
    )
    df.dropna(inplace=True)
    X = df[FEATURE_COLS].values
    y = df["label"].values.astype(int)
    return X, y


def train_xgb(symbol: str, df: pd.DataFrame):
    from xgboost import XGBClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X, y = build_dataset(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"[XGB] {symbol} accuracy: {acc:.3f}")

    save_path = MODELS_PATH / f"{symbol.lower()}_xgb.pkl"
    with open(save_path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)
    print(f"[XGB] Model saved: {save_path}")


class Predictor:
    """Cached model loader — avoids reloading from disk on every call."""
    def __init__(self):
        self._cache = {}

    def predict(self, symbol: str, df: pd.DataFrame) -> float:
        if symbol not in self._cache:
            path = MODELS_PATH / f"{symbol.lower()}_xgb.pkl"
            with open(path, "rb") as f:
                self._cache[symbol] = pickle.load(f)
        bundle = self._cache[symbol]
        if not set(FEATURE_COLS).issubset(df.columns):
            df = add_indicators(df.copy())
        X = df[FEATURE_COLS].iloc[-1:].values
        X_scaled = bundle["scaler"].transform(X)
        prob = bundle["model"].predict_proba(X_scaled)[0][1]
        return round(float(prob), 4)


_PREDICTOR = Predictor()


def predict_xgb(symbol: str, df: pd.DataFrame) -> float:
    return _PREDICTOR.predict(symbol, df)


def train_lstm(df: pd.DataFrame):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import MinMaxScaler

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[LSTM] Using device: {device}")

    df = add_indicators(df.copy())
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[FEATURE_COLS])

    labels = np.where(df["close"].shift(-3).values > df["close"].values * 1.0005, 1, 0)

    X, y = [], []
    for i in range(LSTM_SEQ_LEN, len(scaled) - 3):
        X.append(scaled[i - LSTM_SEQ_LEN:i])
        y.append(labels[i])
    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_t = torch.tensor(X_train)
    y_t = torch.tensor(y_train).unsqueeze(1)
    X_v = torch.tensor(X_test)
    y_v = torch.tensor(y_test).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_t, y_t), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_v, y_v), batch_size=32, shuffle=False)

    class LSTMModel(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.lstm1 = nn.LSTM(input_dim, 64, batch_first=True)
            self.lstm2 = nn.LSTM(64, 32, batch_first=True)
            self.fc1 = nn.Linear(32, 16)
            self.fc2 = nn.Linear(16, 1)
            self.dropout = nn.Dropout(0.2)
            self.relu = nn.ReLU()

        def forward(self, x):
            x, _ = self.lstm1(x)
            x = self.dropout(x)
            x, _ = self.lstm2(x)
            x = self.dropout(x[:, -1, :])
            x = self.relu(self.fc1(x))
            x = torch.sigmoid(self.fc2(x))
            return x

    model = LSTMModel(X.shape[2]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    best_val_loss = float("inf")
    patience = 5
    patience_count = 0

    for epoch in range(50):
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                preds = model(Xb)
                val_loss += criterion(preds, yb).item()
                predicted = (preds > 0.5).float()
                correct += (predicted == yb).sum().item()
                total += yb.size(0)

        val_loss /= len(val_loader)
        acc = correct / total if total > 0 else 0

        if (epoch + 1) % 10 == 0:
            print(f"[LSTM] Epoch {epoch+1:2d}/50  train_loss={train_loss/len(train_loader):.4f}  val_loss={val_loss:.4f}  val_acc={acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            torch.save(model.state_dict(), MODELS_PATH / "xauusd_lstm.pt")
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"[LSTM] Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(MODELS_PATH / "xauusd_lstm.pt"))
    with open(MODELS_PATH / "xauusd_lstm_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"[LSTM] XAU/USD model saved to {MODELS_PATH / 'xauusd_lstm.pt'}")
    print(f"[LSTM] Best val accuracy: {acc:.3f}")


def _lstm_unused(df: pd.DataFrame) -> float:
    import torch
    import torch.nn as nn

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    class LSTMModel(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.lstm1 = nn.LSTM(input_dim, 64, batch_first=True)
            self.lstm2 = nn.LSTM(64, 32, batch_first=True)
            self.fc1 = nn.Linear(32, 16)
            self.fc2 = nn.Linear(16, 1)
            self.dropout = nn.Dropout(0.2)
            self.relu = nn.ReLU()

        def forward(self, x):
            x, _ = self.lstm1(x)
            x = self.dropout(x)
            x, _ = self.lstm2(x)
            x = self.dropout(x[:, -1, :])
            x = self.relu(self.fc1(x))
            x = torch.sigmoid(self.fc2(x))
            return x

    model = LSTMModel(len(FEATURE_COLS)).to(device)
    model.load_state_dict(torch.load(MODELS_PATH / "xauusd_lstm.pt", map_location=device))
    model.eval()

    with open(MODELS_PATH / "xauusd_lstm_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    df = add_indicators(df.copy())
    scaled = scaler.transform(df[FEATURE_COLS])
    seq = scaled[-LSTM_SEQ_LEN:].reshape(1, LSTM_SEQ_LEN, len(FEATURE_COLS))
    seq_t = torch.tensor(seq, dtype=torch.float32).to(device)

    with torch.no_grad():
        prob = float(model(seq_t).cpu().numpy()[0][0])
    return round(prob, 4)


XGB_SYMBOLS = ["BTCUSD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]

def predict(symbol: str, df: pd.DataFrame) -> float:
    if symbol in XGB_SYMBOLS:
        return predict_xgb(symbol, df)
    raise ValueError(f"No model configured for {symbol}")
