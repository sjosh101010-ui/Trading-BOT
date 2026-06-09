"""Generate Backtesting Guide PDF"""
from fpdf import FPDF
from pathlib import Path

OUTPUT = Path(__file__).parent / "Backtesting_Guide.pdf"


class PDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 6, "AI Forex Trading System -- Backtesting Guide", align="C")
            self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(20, 60, 120)
        self.cell(0, 10, title)
        self.ln(4)
        self.set_draw_color(20, 60, 120)
        self.line(self.get_x(), self.get_y(), self.get_x() + 190, self.get_y())
        self.ln(6)

    def sub(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(40, 40, 40)
        self.cell(0, 8, title)
        self.ln(7)

    def text(self, txt):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 6, txt)
        self.ln(2)

    def code(self, txt):
        self.set_fill_color(240, 240, 245)
        self.set_font("Courier", "", 9)
        self.set_text_color(20, 20, 20)
        self.multi_cell(0, 5, txt, fill=True)
        self.ln(3)

    def bullet(self, txt, indent=10):
        self.set_x(self.l_margin + indent)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.cell(5, 6, "-")
        self.multi_cell(0, 6, txt)
        self.ln(1)

    def metric(self, label, value):
        self.set_font("Courier", "", 10)
        self.cell(90, 7, f"  {label}:")
        self.set_font("Courier", "B", 10)
        self.cell(0, 7, value)
        self.ln(7)


pdf = PDF()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()

# Cover
pdf.ln(40)
pdf.set_font("Helvetica", "B", 26)
pdf.set_text_color(20, 60, 120)
pdf.cell(0, 14, "Backtesting Guide", align="C")
pdf.ln(12)
pdf.set_font("Helvetica", "", 16)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 10, "AI Forex Trading System", align="C")
pdf.ln(10)
pdf.set_font("Helvetica", "I", 11)
pdf.set_text_color(100, 100, 100)
pdf.cell(0, 8, "EURUSD M5 Scalper", align="C")
pdf.ln(40)
pdf.set_font("Helvetica", "", 9)
pdf.set_text_color(140, 140, 140)
pdf.cell(0, 6, "June 2026", align="C")

# Page 1: Quick Backtest
pdf.add_page()
pdf.section("1  Running a Backtest")
pdf.text("The quickest way to run a backtest:")
pdf.code("  python3 trade.py backtest")
pdf.text("This runs a cumulative backtest from May 6 to Jun 5 with the")
pdf.text("current config settings (session hours, risk params, etc.).")
pdf.ln(4)

pdf.sub("Custom Date Range")
pdf.text("To backtest a specific period:")
pdf.code("  python3 trade.py backtest --from-date 2026-05-06")
pdf.text("You can also use the raw runner for more control:")
pdf.code("  python3 run_backtest.py \\\n"
        "    --csv data/raw/EURUSD_M5_60d.csv \\\n"
        "    --symbol EURUSD \\\n"
        "    --from-date 2026-05-06 \\\n"
        "    --balance 100")

pdf.sub("Single Day Backtest")
pdf.code("  python3 run_backtest.py \\\n"
        "    --csv data/raw/EURUSD_M5_60d.csv \\\n"
        "    --symbol EURUSD \\\n"
        "    --date 2026-05-15 \\\n"
        "    --balance 100")

# Page 2: Reading Results
pdf.add_page()
pdf.section("2  Reading Backtest Results")
pdf.text("Example output:")
pdf.code(""
"  Results: 83 trades (83 closed)\n"
"  Performance Metrics\n"
"  ========================================\n"
"    Trade Period:      2026-05-06 -> 2026-06-05\n"
"    Total Trades:      83\n"
"    Wins / Losses:     23 / 60\n"
"    Win Rate:          27.7%\n"
"    Profit Factor:     1.53\n"
"    Total P&L:         $+94.34\n"
"    Final Balance:     $194.34\n"
"    Avg Win / Loss:    $+11.83 / $-2.96\n"
"    Max Drawdown:      $22.45 (10.6%)\n"
"    Sharpe Ratio:      6.4\n"
"  ========================================")

pdf.ln(4)
pdf.sub("What Each Metric Means")
pdf.bullet("Win Rate: Percentage of trades that hit TP. Target 30-40% for 1:4 RR.")
pdf.bullet("Profit Factor: Gross profit / gross loss. Above 1.5 is good, above 2.0 is excellent.")
pdf.bullet("Final Balance: Starting $100 + total P&L. Measures absolute return.")
pdf.bullet("Max Drawdown: Largest peak-to-trough decline. Keep under 15%.")
pdf.bullet("Sharpe Ratio: Risk-adjusted return. Above 3 is excellent for forex.")
pdf.bullet("Avg Win / Loss: Average winning and losing trade in dollars.")

pdf.sub("The 1:4 RR Rule")
pdf.text("With 4:1 reward-to-risk, the break-even win rate is 20%.")
pdf.text("At 27.7% WR, expected value per trade = (0.277 x 4) - (0.723 x 1) = +0.39")
pdf.text("The system is profitable as long as WR stays above 20%.")

# Page 3: Comparing Configs
pdf.add_page()
pdf.section("3  Comparing Session Hours")
pdf.text("The session hours config setting has a big impact on results.")
pdf.ln(4)

pdf.set_font("Helvetica", "B", 9)
pdf.set_fill_color(230, 235, 245)
pdf.cell(35, 7, "Session", border=1, fill=True)
pdf.cell(15, 7, "Trades", border=1, fill=True)
pdf.cell(15, 7, "WR", border=1, fill=True)
pdf.cell(15, 7, "PF", border=1, fill=True)
pdf.cell(25, 7, "Final Bal", border=1, fill=True)
pdf.cell(20, 7, "Max DD", border=1, fill=True)
pdf.cell(15, 7, "Sharpe", border=1, fill=True)
pdf.cell(50, 7, "Hours (Manila)", border=1, fill=True)
pdf.ln()

pdf.set_font("Courier", "", 8)
rows = [
    ("03-10 UTC", "53", "37.7%", "2.87", "$281", "5.9%", "15.2", "11AM-6PM"),
    ("03-16 UTC", "83", "27.7%", "1.53", "$194", "10.6%", "6.4", "11AM-12AM"),
]
for r in rows:
    for c in r:
        pdf.cell(35 if r.index(c) == 0 else (15 if r.index(c) in (1,2,3) else (25 if r.index(c) == 4 else (20 if r.index(c) == 5 else (15 if r.index(c) == 6 else 50)))), 7, c, border=1)
    pdf.ln()

pdf.ln(5)
pdf.text("The 03-10 UTC session (11AM-6PM Manila) captures London open")
pdf.text("and early NY overlap. Adding 10-16 UTC (6PM-12AM Manila) dilutes")
pdf.text("quality -- the system's PA patterns work best in the earlier session.")

pdf.ln(4)
pdf.sub("How to Change Session Hours")
pdf.text("Edit config.py:")
pdf.code(""
"  TRADE_SESSION_START = 3     # 03:00 UTC = 11:00 Manila\n"
"  TRADE_SESSION_END   = 10    # 10:00 UTC = 18:00 Manila")
pdf.text("Manila time = UTC + 8. To convert: add 8 to UTC hour.")
pdf.code(""
"  Manila 11AM = 03 UTC   |  Manila  6PM = 10 UTC\n"
"  Manila  8PM = 12 UTC   |  Manila 12AM = 16 UTC")

# Page 4: Advanced
pdf.add_page()
pdf.section("4  Changing Other Settings")
pdf.text("All tunable parameters are in config.py:")
pdf.ln(2)
pdf.code(""
"  BUY_THRESHOLD / SELL_THRESHOLD    # Signal strength required\n"
"  ML_CONFIDENCE_GATE                # ML prob gate (0.15 = need >0.65)\n"
"  BASE_RISK_PCT                     # Risk per trade (0.015 = 1.5%)\n"
"  ATR_MULT: sl: 1.0, tp: 4.0       # SL/TP as ATR multiples\n"
"  MAX_DAILY_TRADES                  # Daily trade limit\n"
"  MAX_DAILY_LOSS_PCT                # Daily loss cap (0.03 = 3%)\n"
"  W_PA, W_ML, W_SENT               # Signal fusion weights\n"
"  XGB_N_ESTIMATORS                  # ML model complexity")

pdf.ln(2)
pdf.sub("Workflow: Change -> Test -> Compare")
pdf.code(""
"  1. Edit config.py (change one setting at a time)\n"
"  2. Run: python3 trade.py backtest\n"
"  3. Note the metrics\n"
"  4. Repeat with different values\n"
"  5. Compare results side by side")

pdf.sub("Common Tweaks to Try")
pdf.bullet("Increase TP multiplier: ATR_MULT tp: 4.0 -> 5.0 (higher RR, lower WR)")
pdf.bullet("Reduce risk: BASE_RISK_PCT 0.015 -> 0.01 (smaller positions, lower DD)")
pdf.bullet("Tighter gate: ML_CONFIDENCE_GATE 0.15 -> 0.20 (fewer trades, higher WR)")
pdf.bullet("Session window: Adjust TRADE_SESSION_START/END (test different hours)")

# Page 5: ML & Walk-Forward
pdf.add_page()
pdf.section("5  ML Model Testing")
pdf.sub("Retrain the Model")
pdf.code("  python3 trade.py train")
pdf.text("Retrains XGBoost on the most recent EURUSD M5 CSV in data/raw/.")
pdf.text("The model predicts if price will move 8+ pips in the next 10 min.")
pdf.text("Current accuracy: 57.8%. Training takes ~30 seconds.")

pdf.sub("Walk-Forward Validation")
pdf.text("Most rigorous test -- no look-ahead bias. Trains on past 30 days,")
pdf.text("tests on next 5 days, slides forward.")
pdf.code("  python3 walk_forward.py")
pdf.text("This runs 7+ independent test windows. All windows profitable in")
pdf.text("the last run confirms the model generalizes to unseen data.")
pdf.ln(4)

pdf.sub("Grid Search (Finding Optimal Params)")
pdf.text("To find the best model settings:")
pdf.code("  python3 grid_search.py")
pdf.text("Tests 25 label combinations (lookahead x threshold) and 28")
pdf.text("hyperparameter groups. Takes ~5 minutes. Results show best")
pdf.text("accuracy, feature importance, and optimal config.")

# Page 6: Quick Reference
pdf.add_page()
pdf.section("6  Quick Command Reference")
pdf.ln(2)
pdf.code(""
"  # Daily use\n"
"  python3 trade.py                  Interactive menu\n"
"  python3 trade.py live             Start live monitor\n"
"  python3 trade.py status           System health check\n"
"  python3 trade.py daily            Full daily routine\n"
"\n"
"  # Backtesting\n"
"  python3 trade.py backtest         Quick backtest (May 6 - Jun 5)\n"
"  python3 run_backtest.py           Full control (see --help)\n"
"    --csv DATA_FILE\n"
"    --symbol EURUSD\n"
"    --date YYYY-MM-DD               Single day\n"
"    --from-date YYYY-MM-DD          Cumulative from date\n"
"    --balance 100                   Starting balance\n"
"\n"
"  # Model & Data\n"
"  python3 trade.py train            Retrain XGBoost model\n"
"  python3 trade.py download         Refresh EURUSD M5 data\n"
"  python3 walk_forward.py           Walk-forward validation\n"
"  python3 grid_search.py            Hyperparameter tuning\n"
"\n"
"  # Config\n"
"  config.py                         All settings in one file\n"
"  logs/backtest_eurusd.csv          Last trade log")
pdf.ln(4)

pdf.sub("Output Files")
pdf.code(""
"  logs/system.log              Full event log\n"
"  logs/backtest_eurusd.csv     Last backtest trades\n"
"  models/eurusd_xgb.pkl       Trained ML model\n"
"  data/raw/EURUSD_M5_*.csv    Market data")

pdf.output(str(OUTPUT))
print(f"PDF saved: {OUTPUT}")
