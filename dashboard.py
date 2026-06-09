"""Streamlit dashboard for the paper trader. Launch with: streamlit run dashboard.py"""
import json, time, os
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Paper Trader", layout="wide")
st.title("BTCUSD M5 Crypto Scalper")

STATE_FILE = "logs/live_state.json"
TRADE_FILE = "logs/paper_trades_live.csv"

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}

def load_trades():
    if os.path.exists(TRADE_FILE):
        df = pd.read_csv(TRADE_FILE)
        if not df.empty and "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
        return df
    return pd.DataFrame()

state = load_state()
trades = load_trades()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Balance", f"${state.get('balance', 0):,.2f}")
col2.metric("Open Positions", str(state.get('open_positions', 0)))
col3.metric("Total Trades", str(len(trades)))
col4.metric("Manila Time", state.get('manila_time', '-'))

if not trades.empty:
    wins = trades[trades["status"] == "WIN"]
    losses = trades[trades["status"] == "LOSS"]
    wr = len(wins) / len(trades) * 100 if len(trades) > 0 else 0
    gp = wins["pnl"].sum() if not wins.empty else 0
    gl = abs(losses["pnl"].sum()) if not losses.empty else 0
    pf = gp / gl if gl > 0 else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Win Rate", f"{wr:.0f}%")
    c2.metric("Profit Factor", f"{pf:.2f}")
    c3.metric("Realized PnL", f"${trades['pnl'].sum():+,.0f}")

    trades_sorted = trades.sort_values("time")
    trades_sorted["cum_pnl"] = trades_sorted["pnl"].cumsum() + 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trades_sorted["time"], y=trades_sorted["cum_pnl"],
                             mode="lines", name="Equity", line=dict(color="green")))
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                      yaxis_title="Balance ($)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent Trades")
    display = trades.tail(20).sort_values("time", ascending=False)
    cols = [c for c in ["time", "symbol", "direction", "pnl", "status", "adx_entry", "duration_hours"] if c in display.columns]
    styled = display[cols].style.applymap(
        lambda v: "color: green" if v == "WIN" else ("color: red" if v == "LOSS" else ""),
        subset=["status"]
    )
    st.dataframe(styled, use_container_width=True)

    st.subheader("PnL Distribution")
    fig2 = px.histogram(trades, x="pnl", nbins=30,
                        color="status", color_discrete_map={"WIN": "green", "LOSS": "red"})
    fig2.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No trades yet. Paper trader running, waiting for signals...")

positions = state.get("positions", {})
if positions:
    st.subheader("Open Positions")
    for sym, p in positions.items():
        st.write(f"**{sym}**: {p['direction']} {p['lot']}L @ {p['entry']}  SL={p['sl']}  TP={p['tp']}")

st.caption(f"Last updated: {state.get('last_update', '-')}")

# Auto-refresh at end (so page renders first)
auto = st.toggle("Auto-refresh (5s)", value=True)
if auto:
    time.sleep(5)
    st.rerun()
