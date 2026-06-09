import numpy as np
import pandas as pd


def calculate_metrics(trades: pd.DataFrame, initial_balance: float = 10000.0) -> dict:
    if trades.empty:
        return {"error": "No trades to analyze"}

    closed = trades[trades["status"].isin(["WIN", "LOSS"])].copy()
    total = len(closed)
    if total == 0:
        return {"error": "No closed trades"}

    trade_start = str(closed["timestamp"].min())[:19]
    trade_end = str(closed["exit_time"].max())[:19]

    wins = (closed["status"] == "WIN").sum()
    losses = (closed["status"] == "LOSS").sum()
    win_rate = wins / total

    total_pnl = closed["pnl_usd"].sum()
    final_balance = initial_balance + total_pnl
    gross_profit = closed[closed["pnl_usd"] > 0]["pnl_usd"].sum()
    gross_loss = abs(closed[closed["pnl_usd"] < 0]["pnl_usd"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_win = closed[closed["pnl_usd"] > 0]["pnl_usd"].mean() if wins > 0 else 0
    avg_loss = closed[closed["pnl_usd"] < 0]["pnl_usd"].mean() if losses > 0 else 0

    closed = closed.sort_values("timestamp")
    closed["balance"] = initial_balance + closed["pnl_usd"].cumsum()
    running_max = closed["balance"].cummax()
    drawdown = running_max - closed["balance"]
    max_drawdown = drawdown.max()
    max_drawdown_pct = max_drawdown / running_max.max() if running_max.max() > 0 else 0

    avg_bars_held = 0
    if "bars_held" in closed.columns:
        avg_bars_held = closed["bars_held"].mean()
    elif "entry_time" in closed.columns and "exit_time" in closed.columns:
        avg_bars_held = (pd.to_datetime(closed["exit_time"]) - pd.to_datetime(closed["entry_time"])).dt.total_seconds().mean() / 300

    returns = closed["pnl_usd"] / initial_balance
    sharpe = np.nan
    if len(returns) > 1 and returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(252 * 6)

    return {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 2),
        "total_pnl": round(total_pnl, 2),
        "final_balance": round(final_balance, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "max_drawdown": round(max_drawdown, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 4),
        "avg_bars_held": round(avg_bars_held, 1) if avg_bars_held > 0 else 0,
        "sharpe_ratio": round(sharpe, 2) if not np.isnan(sharpe) else None,
        "trade_start": trade_start,
        "trade_end": trade_end,
    }


def print_metrics(metrics: dict):
    if "error" in metrics:
        print(f"  {metrics['error']}")
        return
    print("  Performance Metrics")
    print("=" * 40)
    print(f"  Trade Period:      {metrics['trade_start']}  ->  {metrics['trade_end']}")
    print(f"  Total Trades:      {metrics['total_trades']}")
    print(f"  Wins / Losses:     {metrics['wins']} / {metrics['losses']}")
    print(f"  Win Rate:          {metrics['win_rate']:.1%}")
    print(f"  Profit Factor:     {metrics['profit_factor']}")
    print(f"  Total P&L:         ${metrics['total_pnl']:+.2f}")
    print(f"  Final Balance:     ${metrics['final_balance']:.2f}")
    print(f"  Avg Win / Loss:    ${metrics['avg_win']:+.2f} / ${metrics['avg_loss']:+.2f}")
    print(f"  Max Drawdown:      ${metrics['max_drawdown']:.2f} ({metrics['max_drawdown_pct']:.1%})")
    if metrics["avg_bars_held"]:
        print(f"  Avg Bars Held:     {metrics['avg_bars_held']}")
    if metrics["sharpe_ratio"] is not None:
        print(f"  Sharpe Ratio:      {metrics['sharpe_ratio']}")
    print("=" * 40)
