import pytest
import pandas as pd
import numpy as np
from analysis.feature_engineering import (
    add_indicators, compute_trend_score, ema, rsi, macd, atr, bbands,
    stoch, adx, cci, williams_r, mfi, ema_slope, autocorr,
)


@pytest.fixture
def sample_df():
    idx = pd.date_range("2024-01-01", periods=200, freq="5min")
    close = np.cumsum(np.random.randn(200) * 0.0005) + 1.08
    high = close * (1 + np.abs(np.random.randn(200) * 0.001))
    low = close * (1 - np.abs(np.random.randn(200) * 0.001))
    open_ = (high + low) / 2
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": np.random.randint(100, 1000, 200),
    }, index=idx)


class TestAddIndicators:
    def test_all_columns_present(self, sample_df):
        result = add_indicators(sample_df)
        expected = {
            "ema_8", "ema_21", "ema_50", "rsi", "macd", "macd_sig",
            "atr", "bb_upper", "bb_lower", "bb_mid", "stoch_k", "stoch_d",
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
        }
        assert expected.issubset(set(result.columns)), \
            f"Missing columns: {expected - set(result.columns)}"

    def test_no_nans_in_result(self, sample_df):
        result = add_indicators(sample_df)
        assert not result.isna().any().any(), "NaN values remain in output"

    def test_shorter_output(self, sample_df):
        input_len = len(sample_df)
        result = add_indicators(sample_df)
        assert len(result) < input_len, "Should drop NaN rows"

    def test_preserves_ohlcv(self, sample_df):
        result = add_indicators(sample_df)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns


class TestComputeTrendScore:
    def test_uptrend_positive(self):
        idx = pd.date_range("2024-01-01", periods=100, freq="5min")
        close = np.linspace(1.07, 1.09, 100)
        df = pd.DataFrame({
            "close": close, "open": close * 0.999, "high": close * 1.001,
            "low": close * 0.999, "volume": 500,
            "ema_8": np.linspace(1.072, 1.088, 100),
            "ema_21": np.linspace(1.071, 1.085, 100),
            "ema_50": np.linspace(1.070, 1.080, 100),
            "rsi": np.linspace(50, 65, 100),
        }, index=idx)
        score = compute_trend_score(df)
        assert score > 0

    def test_downtrend_negative(self):
        idx = pd.date_range("2024-01-01", periods=100, freq="5min")
        close = np.linspace(1.09, 1.07, 100)
        df = pd.DataFrame({
            "close": close, "open": close * 1.001, "high": close * 1.002,
            "low": close * 0.999, "volume": 500,
            "ema_8": np.linspace(1.088, 1.072, 100),
            "ema_21": np.linspace(1.085, 1.071, 100),
            "ema_50": np.linspace(1.080, 1.070, 100),
            "rsi": np.linspace(65, 35, 100),
        }, index=idx)
        score = compute_trend_score(df)
        assert score < 0

    def test_insufficient_data_returns_zero(self):
        df = pd.DataFrame({"close": [1.08] * 10, "ema_50": [1.08] * 10,
                           "ema_21": [1.08] * 10, "ema_8": [1.08] * 10,
                           "rsi": [50] * 10})
        score = compute_trend_score(df)
        assert score == 0.0


class TestIndividualIndicators:
    def test_ema(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ema(s, 3)
        assert len(result) == 5
        assert not result.isna().all()

    def test_rsi(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 2.0,
                       3.0, 4.0, 5.0, 6.0, 7.0])
        result = rsi(s, 5)
        assert len(result) == 15
        assert 0 <= result.iloc[-1] <= 100

    def test_macd(self):
        s = pd.Series(np.random.randn(50) * 0.01 + 1.08)
        result = macd(s, 5, 10, 3)
        assert "MACD" in result.columns
        assert "MACDs" in result.columns

    def test_atr(self):
        high = pd.Series(np.random.randn(50) * 0.001 + 1.081)
        low = pd.Series(np.random.randn(50) * 0.001 + 1.079)
        close = pd.Series(np.random.randn(50) * 0.001 + 1.08)
        result = atr(high, low, close, 5)
        assert len(result) == 50
        assert result.iloc[-1] > 0

    def test_bbands(self):
        s = pd.Series(np.random.randn(50) * 0.01 + 1.08)
        result = bbands(s, 5, 2)
        for col in ["BBU", "BBM", "BBL"]:
            assert col in result.columns
        valid = result.dropna()
        assert (valid["BBU"] >= valid["BBM"]).all()
        assert (valid["BBL"] <= valid["BBM"]).all()

    def test_stoch(self):
        high = pd.Series(np.random.randn(50) * 0.001 + 1.082)
        low = pd.Series(np.random.randn(50) * 0.001 + 1.078)
        close = pd.Series(np.random.randn(50) * 0.001 + 1.08)
        result = stoch(high, low, close, 5, 3)
        assert "STOCHk" in result.columns
        assert "STOCHd" in result.columns

    def test_adx(self):
        high = pd.Series(np.random.randn(100) * 0.001 + 1.081)
        low = pd.Series(np.random.randn(100) * 0.001 + 1.079)
        close = pd.Series(np.random.randn(100) * 0.001 + 1.08)
        result = adx(high, low, close, 5)
        assert len(result) == 100
        assert 0 <= result.iloc[-1] <= 100

    def test_cci(self):
        high = pd.Series(np.random.randn(50) * 0.001 + 1.081)
        low = pd.Series(np.random.randn(50) * 0.001 + 1.079)
        close = pd.Series(np.random.randn(50) * 0.001 + 1.08)
        result = cci(high, low, close, 5)
        assert len(result) == 50

    def test_williams_r(self):
        high = pd.Series(np.random.randn(50) * 0.001 + 1.082)
        low = pd.Series(np.random.randn(50) * 0.001 + 1.078)
        close = pd.Series(np.random.randn(50) * 0.001 + 1.08)
        result = williams_r(high, low, close, 5)
        assert -100 <= result.iloc[-1] <= 0

    def test_mfi(self):
        high = pd.Series(np.random.randn(50) * 0.001 + 1.081)
        low = pd.Series(np.random.randn(50) * 0.001 + 1.079)
        close = pd.Series(np.random.randn(50) * 0.001 + 1.08)
        volume = pd.Series(np.random.randint(100, 1000, 50))
        result = mfi(high, low, close, volume, 5)
        assert len(result) == 50
        assert 0 <= result.iloc[-1] <= 100

    def test_ema_slope(self):
        s = pd.Series(np.linspace(1.0, 2.0, 50))
        result = ema_slope(s, 3)
        assert len(result) == 50
        assert result.iloc[-1] > 0

    def test_autocorr(self):
        s = pd.Series(np.random.randn(100))
        result = autocorr(s, lag=1, window=10)
        assert len(result) == 100
        assert -1 <= result.iloc[-1] <= 1
