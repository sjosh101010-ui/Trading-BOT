"""Generate AI Forex Trading System -- PDF User Guide"""
from fpdf import FPDF
from pathlib import Path

OUTPUT = Path(__file__).parent / "AI_Forex_Trading_Guide.pdf"


class GuidePDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 6, "AI Forex Trading System -- EURUSD Scalper", align="C")
            self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(20, 60, 120)
        self.cell(0, 10, title)
        self.ln(4)
        self.set_draw_color(20, 60, 120)
        self.line(self.get_x(), self.get_y(), self.get_x() + 190, self.get_y())
        self.ln(6)

    def subsection(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(40, 40, 40)
        self.cell(0, 8, title)
        self.ln(7)

    def body_text(self, text):
        self.set_font("Courier", "", 9)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def body_regular(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def code_block(self, text):
        self.set_fill_color(240, 240, 245)
        self.set_font("Courier", "", 9)
        self.set_text_color(20, 20, 20)
        self.multi_cell(0, 5, text, fill=True)
        self.ln(3)

    def bullet(self, text, indent=10):
        self.set_x(self.l_margin + indent)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.cell(5, 6, "-")
        self.multi_cell(0, 6, text)
        self.ln(1)

    def table_row(self, cells, bold=False, fill=False):
        self.set_font("Helvetica", "B" if bold else "", 9)
        if fill:
            self.set_fill_color(230, 235, 245)
        widths = [40, 150]
        for i, cell in enumerate(cells):
            self.cell(widths[i], 7, cell, border=1, fill=fill)
        self.ln()


pdf = GuidePDF()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()

# == Cover ==
pdf.ln(40)
pdf.set_font("Helvetica", "B", 26)
pdf.set_text_color(20, 60, 120)
pdf.cell(0, 14, "AI Forex Trading System", align="C")
pdf.ln(12)
pdf.set_font("Helvetica", "", 16)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 10, "EURUSD M5 Scalper", align="C")
pdf.ln(10)
pdf.set_font("Helvetica", "I", 11)
pdf.set_text_color(100, 100, 100)
pdf.cell(0, 8, "Daily User Guide", align="C")
pdf.ln(6)
pdf.cell(0, 8, "Session: 11:00 - 18:00 Manila  |  03:00 - 10:00 UTC", align="C")
pdf.ln(40)
pdf.set_font("Helvetica", "", 9)
pdf.set_text_color(140, 140, 140)
pdf.cell(0, 6, "Generated: June 2026", align="C")

# == Page 2: Quick Start ==
pdf.add_page()
pdf.section_title("1  Quick Start")
pdf.body_regular("Run the system with a single command before your trading session:")
pdf.code_block("  cd ~/Personal/Trade\n  python3 trade.py live")
pdf.body_regular("That's it. The live monitor opens a dashboard that refreshes every 60")
pdf.body_regular("seconds, fetching real EURUSD data from Yahoo Finance.")
pdf.ln(4)
pdf.body_regular("If you prefer an interactive menu:")
pdf.code_block("  python3 trade.py")
pdf.ln(4)

pdf.subsection("All Commands")
pdf.code_block(
    "  python3 trade.py              Interactive menu\n"
    "  python3 trade.py live         Start live monitor (main use)\n"
    "  python3 trade.py status       System health check\n"
    "  python3 trade.py daily        Full daily routine\n"
    "  python3 trade.py backtest     Run backtest\n"
    "  python3 trade.py train        Retrain ML model\n"
    "  python3 trade.py download     Refresh EURUSD data"
)

# == Daily Workflow ==
pdf.add_page()
pdf.section_title("2  Daily Workflow")
pdf.ln(2)

pdf.subsection("Morning Setup (10:55 AM Manila)")
pdf.body_regular("Open a terminal and run the live monitor. It will wait until 11:00 AM")
pdf.body_regular("to start scanning for signals.")
pdf.code_block("  python3 trade.py live")

pdf.subsection("During the Session (11:00 AM - 6:00 PM)")
pdf.body_regular("The dashboard updates every 60 seconds. Watch for signal arrows:")
pdf.ln(2)

# Signal table
pdf.set_font("Helvetica", "B", 9)
pdf.set_fill_color(230, 235, 245)
pdf.table_row(["Signal", "Meaning"], bold=True, fill=True)
pdf.set_font("Helvetica", "", 9)
pdf.table_row(["  ^ BUY", "Entry signal fired - price predicted to rise"])
pdf.table_row(["  v SELL", "Entry signal fired - price predicted to fall"])
pdf.table_row(["  - SKIP", "No signal - continue watching"])
pdf.ln(4)

pdf.subsection("End of Day (6:00 PM)")
pdf.body_regular("The session auto-closes. Press Ctrl+C to stop the monitor.")
pdf.body_regular("Optionally run the end-of-day check:")
pdf.code_block("  python3 trade.py daily")

# == Understanding the Dashboard ==
pdf.add_page()
pdf.section_title("3  Understanding the Dashboard")
pdf.ln(2)

pdf.code_block(
    "  ==========================================================\n"
    "    EURUSD Scalper  |  14:25 Manila  |  Session: ACTIVE\n"
    "  ==========================================================\n"
    "    Signal:  ^ BUY\n"
    "    Score:   +0.623  (PA=+0.75, ML=0.710, conf=62%)\n"
    "    Entry:   1.08455\n"
    "    SL/TP:   1.08395 / 1.08755\n"
    "    Lot:     0.025\n"
    "  \n"
    "    Positions: 1 open\n"
    "      BUY 0.025L @ 1.08455  SL:1.08395 TP:1.08755\n"
    "  \n"
    "    Balance:  $103.47\n"
    "    Today:   +$2.14 (+2.1%)  |  Total: +$3.47 (+3.5%)\n"
    "    Trades:  1/8 today  |  Refresh: 60s\n"
    "  =========================================================="
)

pdf.ln(4)
pdf.subsection("Dashboard Fields")
pdf.bullet("Signal: Entry direction (BUY / SELL / SKIP)")
pdf.bullet("Score: Combined signal strength (PA + ML + Sentiment)")
pdf.bullet("PA: Price action score from -1 (bearish) to +1 (bullish)")
pdf.bullet("ML: ML model confidence from 0.0 to 1.0. Gate requires >0.65")
pdf.bullet("Entry: Suggested entry price")
pdf.bullet("SL/TP: Stop loss and take profit levels")
pdf.bullet("Lot: Position size in lots (based on risk)")
pdf.bullet("Balance: Simulated account balance")
pdf.bullet("Today: Today's P&L and percentage")
pdf.bullet("Trades: Trade count today vs daily limit (8)")

# == How Signals Work ==
pdf.add_page()
pdf.section_title("4  How Signals Work")
pdf.ln(2)

pdf.body_regular("The system uses a three-component fusion to generate signals:")
pdf.ln(2)

pdf.subsection("1. Price Action (70% weight)")
pdf.body_regular("Scans M5 candles for RSI extremes, Bollinger Band touches,")
pdf.body_regular("EMA pullbacks in trend, and range breakouts. This is the")
pdf.body_regular("primary signal driver at 70% weight.")

pdf.subsection("2. ML Model (15% weight)")
pdf.body_regular("XGBoost classifier trained on 16,927 M5 candles. Predicts if")
pdf.body_regular("price will move 8+ pips in the next 10 minutes. Current accuracy:")
pdf.body_regular("57.8%. A confidence gate (ML > 0.65 or < 0.35) blocks low-confidence")
pdf.body_regular("signals to improve win rate.")

pdf.subsection("3. Sentiment (15% weight)")
pdf.body_regular("News headline sentiment via FinBERT. Currently 0 in simulator.")
pdf.body_regular("Can be activated with a news feed.")

pdf.subsection("Signal Flow")
pdf.code_block(
    "  M5 Candle --> PA Score -->+ \n"
    "  ML Model ---> ML Prob -->+--> Fuse --> Signal\n"
    "  Sentiment --> Score ----->+         (BUY/SELL/SKIP)\n"
    "                                        |\n"
    "                                  Confidence Gate\n"
    "                                  (requires ML>0.65)\n"
    "                                        |\n"
    "                                  Trade Execution\n"
    "                                  (with SL/TP 1:4)"
)

# == Risk Management ==
pdf.add_page()
pdf.section_title("5  Risk Management")
pdf.ln(2)

pdf.body_regular("All risk controls are built-in and automatic:")
pdf.ln(2)

pdf.subsection("Fixed Rules")
pdf.bullet("Max 8 trades per day")
pdf.bullet("Max 3% daily loss cap (auto-stops trading)")
pdf.bullet("Risk per trade: 1.5% of balance (scales with confidence up to 3%)")
pdf.bullet("Stop loss: 1.0 x ATR (typically 8-12 pips)")
pdf.bullet("Take profit: 4.0 x ATR (1:4 reward-to-risk)")
pdf.bullet("Session-restricted: 03:00 - 10:00 UTC only")
pdf.bullet("EURUSD only - single pair focus")

pdf.subsection("Why 1:4 RR?")
pdf.body_regular("With a 37.7% win rate and 4:1 RR, the expected value per trade is:")
pdf.code_block(
    "  EV = (0.377 x 4) - (0.623 x 1) = 1.508 - 0.623 = +0.885 units\n"
    "  This means every dollar risked returns $0.885 on average."
)
pdf.body_regular("Even at 25% win rate, the system breaks even with 4:1 RR.")

pdf.subsection("Position Sizing")
pdf.body_regular("Lot size = Risk_USD / (SL_pips x $10_per_pip_per_lot)")
pdf.ln(2)
pdf.body_regular("Example: $100 balance, 1.5% risk ($1.50), 10 pip SL")
pdf.code_block("  Lot = $1.50 / (10 x $10) = 0.015 lots => rounds to 0.01 min")

# == Performance ==
pdf.add_page()
pdf.section_title("6  Performance Summary")
pdf.ln(2)

pdf.subsection("Backtest: $100 -> May 6 -> Jun 5 (22 trading days)")
pdf.ln(2)
pdf.set_font("Courier", "", 10)
pdf.set_fill_color(235, 245, 235)
pdf.cell(0, 7, "  Win Rate:          37.7%", fill=True)
pdf.ln(7)
pdf.cell(0, 7, "  Profit Factor:     2.87", fill=True)
pdf.ln(7)
pdf.cell(0, 7, "  Final Balance:     $280.65 (+180.7%)", fill=True)
pdf.ln(7)
pdf.cell(0, 7, "  Total Trades:      53", fill=True)
pdf.ln(7)
pdf.cell(0, 7, "  Max Drawdown:      5.9% ($16.54)", fill=True)
pdf.ln(7)
pdf.cell(0, 7, "  Sharpe Ratio:      15.22", fill=True)
pdf.ln(10)

pdf.subsection("Walk-Forward (Out-of-Sample)")
pdf.body_regular("7 independent 5-day windows, each trained on prior 30 days only.")
pdf.body_regular("Zero look-ahead bias. All 7 windows were profitable.")
pdf.ln(2)
pdf.set_font("Courier", "", 10)
pdf.set_fill_color(235, 245, 235)
pdf.cell(0, 7, "  Win Rate:          41.2%", fill=True)
pdf.ln(7)
pdf.cell(0, 7, "  Final Balance:     $219.08 (+119%)", fill=True)
pdf.ln(7)
pdf.cell(0, 7, "  Profitable Windows: 7/7 (100%)", fill=True)
pdf.ln(10)

pdf.subsection("ML Model Accuracy")
pdf.set_font("Courier", "", 10)
pdf.cell(0, 7, "  Original (14d training):    34.5%")
pdf.ln(7)
pdf.cell(0, 7, "  Improved (60d + grid search): 57.8%")
pdf.ln(7)
pdf.cell(0, 7, "  Features: 28 indicators (17 base + 11 added)")
pdf.ln(7)
pdf.cell(0, 7, "  Best params: lookahead=2, threshold=8 pips,")
pdf.cell(0, 7, "  lr=0.01, depth=5, n_est=100, gamma=0.1")

# == Files & Maintenance ==
pdf.add_page()
pdf.section_title("7  Files & Maintenance")
pdf.ln(2)

pdf.subsection("Project Structure")
pdf.code_block(
    "  trade.py              CLI launcher (main entry point)\n"
    "  main.py               Live monitor (yfinance data)\n"
    "  config.py             All settings in one place\n"
    "  train_models.py       ML model training\n"
    "  run_backtest.py        Backtest runner\n"
    "  walk_forward.py       Walk-forward validation\n"
    "  grid_search.py        Hyperparameter tuning\n"
    "  data/raw/             EURUSD M5 CSV data files\n"
    "  models/               Trained XGBoost model\n"
    "  logs/                 Log files and trade records"
)

pdf.subsection("Regular Maintenance")
pdf.bullet("Weekly: python3 trade.py download (refresh data)")
pdf.bullet("Every 2 weeks: python3 trade.py train (retrain model)")
pdf.bullet("After retraining: python3 trade.py backtest (verify)")
pdf.bullet("Check logs/system.log if something seems wrong")

pdf.subsection("Configuration (config.py)")
pdf.body_regular("Key parameters you can tune:")
pdf.code_block(
    "  BUY_THRESHOLD / SELL_THRESHOLD   Signal entry threshold\n"
    "  ML_CONFIDENCE_GATE               ML confidence gate\n"
    "  BASE_RISK_PCT                    Risk per trade\n"
    "  ATR_MULT                         SL/TP multipliers\n"
    "  MAX_DAILY_TRADES                 Daily trade limit\n"
    "  MAX_DAILY_LOSS_PCT               Daily loss cap\n"
    "  TRADE_SESSION_START/END          Trading hours UTC"
)

pdf.subsection("Important Notes")
pdf.bullet("System uses SIMULATOR mode (no real money)")
pdf.bullet("Yahoo Finance data has 0 volume for forex (clipped to 1)")
pdf.bullet("Session: 03:00-10:00 UTC = 11:00-18:00 Manila time")
pdf.bullet("MT5 not available on macOS - simulator only")
pdf.bullet("Best for: EURUSD scalping with 1:4 RR during active hours")

pdf.output(str(OUTPUT))
print(f"PDF saved: {OUTPUT}")
