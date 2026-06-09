import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data.storage import get_connection
from datetime import datetime

st.set_page_config(page_title="AI Forex Bot", page_icon="📈", layout="wide")


def load_trades() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 200", conn)


def load_signals() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql("SELECT * FROM signals ORDER BY timestamp DESC LIMIT 50", conn)


st.title("AI Forex Scalping Bot — Live Dashboard")
st.caption(f"Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")

trades = load_trades()
if not trades.empty:
    closed = trades[trades["status"].isin(["WIN", "LOSS"])]
    wins = (closed["status"] == "WIN").sum()
    total = len(closed)
    win_rate = wins / total * 100 if total > 0 else 0
    total_pnl = closed["pnl_usd"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Trades", total)
    c2.metric("Win Rate", f"{win_rate:.1f}%")
    c3.metric("Total P&L", f"${total_pnl:+.2f}")
    c4.metric("Open Positions", trades[trades["status"] == "OPEN"]["symbol"].nunique())

st.subheader("Equity curve")
if not trades.empty:
    closed = trades[trades["status"].isin(["WIN", "LOSS"])].sort_values("timestamp")
    closed["cumulative_pnl"] = closed["pnl_usd"].cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=closed["timestamp"],
        y=closed["cumulative_pnl"],
        mode="lines+markers",
        line=dict(color="#1B5E20", width=2),
        name="Equity",
    ))
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

col_left, col_right = st.columns(2)
with col_left:
    st.subheader("Recent signals")
    signals = load_signals()
    if not signals.empty:
        st.dataframe(signals[["timestamp", "symbol", "final_signal", "final_score",
                              "pa_score", "ml_score", "sentiment_score"]]
                     .head(15), use_container_width=True)

with col_right:
    st.subheader("Trade history")
    if not trades.empty:
        st.dataframe(trades[["timestamp", "symbol", "direction", "lot_size",
                             "pnl_usd", "status"]]
                     .head(15), use_container_width=True)

import time
time.sleep(30)
st.rerun()
