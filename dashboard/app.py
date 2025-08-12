import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import text
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.db import get_engine, ensure_schema

# === Setup ===
st.set_page_config(page_title="QuantFlow", layout="wide")
st.title("📈 QuantFlow Dashboard")

engine = get_engine()
ensure_schema(engine)

# --- Create alert_rules table if not exists ---
ALERT_RULES_DDL = """
CREATE TABLE IF NOT EXISTS alert_rules (
  "Symbol" TEXT PRIMARY KEY,
  "rsi_low" DOUBLE PRECISION DEFAULT 30,
  "rsi_high" DOUBLE PRECISION DEFAULT 70,
  "move_1d_abs_gt" DOUBLE PRECISION DEFAULT 3.0,
  "gap_abs_gt" DOUBLE PRECISION DEFAULT 2.0,
  "near_52w_high_within" DOUBLE PRECISION DEFAULT 1.0,
  "near_52w_low_within" DOUBLE PRECISION DEFAULT 1.0,
  "updated_at" TIMESTAMP DEFAULT NOW()
);
"""
with engine.begin() as conn:
    conn.execute(text(ALERT_RULES_DDL))

TABLE_NAME = "stocks"

# === Symbols ===
default_symbols = [
    # US Mega/Large
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","AVGO","BRK-B","JPM",
    "V","MA","HD","UNH","LLY","XOM","JNJ","PG","PEP","COST",
    # US Tech & Semi
    "ORCL","CRM","ADBE","AMD","INTC","QCOM","TXN","AMAT","MU","INTU",
    # US Other Sectors
    "WMT","KO","MCD","DIS","NKE","PFE","T","VZ","BAC","GS",
    # Europe
    "ASML","SAP","MC.PA","AI.PA","ORA.PA","SAN.PA","BN.PA","NESN.SW","ROG.SW","NOVN.SW",
    # UK (LSE)
    "AZN.L","SHEL.L","ULVR.L","HSBA.L","BP.L","GSK.L","RIO.L","BATS.L","DGE.L","LSEG.L",
    # Asia/ADRs
    "BABA","TSM","NTES","JD","UBER","SHOP","SQ","SE"
]
symbol = st.selectbox("📌 Select a stock", default_symbols, index=0)
custom_symbol = st.text_input("Or enter another symbol:", "")
if custom_symbol.strip():
    symbol = custom_symbol.upper()

# === Date Filter ===
st.sidebar.header("📅 Date Range")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2024-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# ===== DB Helpers =====
def load_from_db(sym: str) -> pd.DataFrame:
    q = text('SELECT * FROM stocks WHERE "Symbol" = :s ORDER BY "Date"')
    return pd.read_sql(q, engine, params={"s": sym.upper()})

def fetch_live_and_save(sym: str, period="6mo"):
    df = yf.download(sym, period=period, interval="1d")
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    if "Adj Close" in df.columns:
        df.rename(columns={"Adj Close": "Adj_Close"}, inplace=True)
    else:
        df["Adj_Close"] = df["Close"]
    df["Symbol"] = sym.upper()
    df = df[["Date","Open","High","Low","Close","Adj_Close","Volume","Symbol"]]
    df.to_sql(TABLE_NAME, con=engine, if_exists="append", index=False, method="multi", chunksize=1000)
    return df

def get_or_fetch(sym: str) -> pd.DataFrame:
    df = load_from_db(sym)
    if df.empty:
        df = fetch_live_and_save(sym, period="1y")
    return df

# ===== Alerts: Compute signals + rules =====
import numpy as np
def _rolling_extrema(close, high, low, lb=252):
    look = min(len(close), lb)
    if look < 1:
        return None, None
    return high.iloc[-look:].max(), low.iloc[-look:].min()

def compute_signals_df(df: pd.DataFrame) -> dict:
    df = df.sort_values("Date").reset_index(drop=True)
    close = df["Close"].astype(float)
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)
    openp = df["Open"].astype(float)

    # RSI
    rsi_series = RSIIndicator(close=close).rsi() if len(close) >= 15 else pd.Series([np.nan]*len(close))
    rsi_now = rsi_series.iloc[-1]
    rsi_prev = rsi_series.iloc[-2] if len(rsi_series) >= 2 else np.nan

    # SMAs
    sma20_series = SMAIndicator(close=close, window=20).sma_indicator() if len(close) >= 20 else pd.Series([np.nan]*len(close))
    sma50_series = SMAIndicator(close=close, window=50).sma_indicator() if len(close) >= 50 else pd.Series([np.nan]*len(close))
    s20_now = sma20_series.iloc[-1]
    s50_now = sma50_series.iloc[-1]
    s20_prev = sma20_series.iloc[-2] if len(sma20_series) >= 2 else np.nan
    s50_prev = sma50_series.iloc[-2] if len(sma50_series) >= 2 else np.nan

    # 52w
    h52, l52 = _rolling_extrema(close, high, low, lb=252)
    last = close.iloc[-1]

    # 1D move & gap
    ret_1d = (last / close.iloc[-2] - 1.0) * 100.0 if len(close) >= 2 else np.nan
    gap_pct = (openp.iloc[-1] / close.iloc[-2] - 1.0) * 100.0 if len(openp) >= 2 and len(close) >= 2 else np.nan

    return {
        "rsi": None if np.isnan(rsi_now) else round(float(rsi_now), 2),
        "rsi_prev": None if np.isnan(rsi_prev) else round(float(rsi_prev), 2),
        "sma20": None if np.isnan(s20_now) else round(float(s20_now), 4),
        "sma50": None if np.isnan(s50_now) else round(float(s50_now), 4),
        "golden_cross_today": (s20_prev < s50_prev) and (s20_now >= s50_now) if not any(np.isnan([s20_prev,s50_prev,s20_now,s50_now])) else False,
        "death_cross_today":  (s20_prev > s50_prev) and (s20_now <= s50_now) if not any(np.isnan([s20_prev,s50_prev,s20_now,s50_now])) else False,
        "h52": None if h52 is None else float(h52),
        "l52": None if l52 is None else float(l52),
        "pct_to_52w_high": None if not h52 else round((last / h52 - 1.0) * 100.0, 2),
        "pct_to_52w_low":  None if not l52 else round((last / l52 - 1.0) * 100.0, 2),
        "ret_1d_pct": None if np.isnan(ret_1d) else round(float(ret_1d), 2),
        "gap_open_pct": None if np.isnan(gap_pct) else round(float(gap_pct), 2),
        "last": round(float(last), 4),
    }

def default_rules():
    return {
        "rsi_low": 30.0,
        "rsi_high": 70.0,
        "move_1d_abs_gt": 3.0,
        "gap_abs_gt": 2.0,
        "near_52w_high_within": 1.0,
        "near_52w_low_within": 1.0,
    }

def load_rules(sym: str):
    q = text('SELECT * FROM alert_rules WHERE "Symbol" = :s')
    df = pd.read_sql(q, engine, params={"s": sym.upper()})
    if df.empty:
        return default_rules()
    row = df.iloc[0].to_dict()
    return {
        "rsi_low": float(row.get("rsi_low", 30)),
        "rsi_high": float(row.get("rsi_high", 70)),
        "move_1d_abs_gt": float(row.get("move_1d_abs_gt", 3)),
        "gap_abs_gt": float(row.get("gap_abs_gt", 2)),
        "near_52w_high_within": float(row.get("near_52w_high_within", 1)),
        "near_52w_low_within": float(row.get("near_52w_low_within", 1)),
    }

def save_rules(sym: str, rules: dict):
    symu = sym.upper()
    # Upsert: Postgres ON CONFLICT; fallback: delete+insert
    if engine.dialect.name == "postgresql":
        up = text("""
            INSERT INTO alert_rules ("Symbol","rsi_low","rsi_high","move_1d_abs_gt","gap_abs_gt","near_52w_high_within","near_52w_low_within","updated_at")
            VALUES (:s,:rsi_low,:rsi_high,:m1d,:gap,:nh,:nl, NOW())
            ON CONFLICT ("Symbol") DO UPDATE SET
              "rsi_low"=EXCLUDED."rsi_low",
              "rsi_high"=EXCLUDED."rsi_high",
              "move_1d_abs_gt"=EXCLUDED."move_1d_abs_gt",
              "gap_abs_gt"=EXCLUDED."gap_abs_gt",
              "near_52w_high_within"=EXCLUDED."near_52w_high_within",
              "near_52w_low_within"=EXCLUDED."near_52w_low_within",
              "updated_at"=NOW();
        """)
        with engine.begin() as conn:
            conn.execute(up, dict(
                s=symu, rsi_low=rules["rsi_low"], rsi_high=rules["rsi_high"],
                m1d=rules["move_1d_abs_gt"], gap=rules["gap_abs_gt"],
                nh=rules["near_52w_high_within"], nl=rules["near_52w_low_within"]
            ))
    else:
        with engine.begin() as conn:
            conn.execute(text('DELETE FROM alert_rules WHERE "Symbol" = :s'), {"s": symu})
            conn.execute(text("""
                INSERT INTO alert_rules ("Symbol","rsi_low","rsi_high","move_1d_abs_gt","gap_abs_gt","near_52w_high_within","near_52w_low_within","updated_at")
                VALUES (:s,:rsi_low,:rsi_high,:m1d,:gap,:nh,:nl, CURRENT_TIMESTAMP)
            """), dict(
                s=symu, rsi_low=rules["rsi_low"], rsi_high=rules["rsi_high"],
                m1d=rules["move_1d_abs_gt"], gap=rules["gap_abs_gt"],
                nh=rules["near_52w_high_within"], nl=rules["near_52w_low_within"]
            ))

def apply_alerts(signals: dict, rules: dict):
    alerts = []
    rsi = signals.get("rsi")
    if rsi is not None:
        if rsi <= rules["rsi_low"]:
            alerts.append(f"RSI ≤ {rules['rsi_low']} (oversold) [{rsi}]")
        if rsi >= rules["rsi_high"]:
            alerts.append(f"RSI ≥ {rules['rsi_high']} (overbought) [{rsi}]")
    if signals.get("golden_cross_today"): alerts.append("Golden cross today (SMA20 ↑ over SMA50)")
    if signals.get("death_cross_today"):  alerts.append("Death cross today (SMA20 ↓ under SMA50)")

    m1d = signals.get("ret_1d_pct")
    if m1d is not None and abs(m1d) >= rules["move_1d_abs_gt"]:
        alerts.append(f"1D move ≥ {rules['move_1d_abs_gt']}% [{m1d}%]")

    gap = signals.get("gap_open_pct")
    if gap is not None and abs(gap) >= rules["gap_abs_gt"]:
        alerts.append(f"Gap open ≥ {rules['gap_abs_gt']}% [{gap}%]")

    p2h = signals.get("pct_to_52w_high")
    if p2h is not None and p2h >= -rules["near_52w_high_within"] and p2h <= 0:
        alerts.append(f"Within {rules['near_52w_high_within']}% of 52w high [{p2h}%]")
    p2l = signals.get("pct_to_52w_low")
    if p2l is not None and p2l <= rules["near_52w_low_within"] and p2l >= 0:
        alerts.append(f"Within {rules['near_52w_low_within']}% of 52w low [{p2l}%]")

    if signals.get("breakout_52w_high"): alerts.append("New 52w high breakout")
    if signals.get("breakdown_52w_low"): alerts.append("New 52w low breakdown")

    return alerts

# === Fetch/Load data for the selected symbol ===
df = get_or_fetch(symbol)
if df.empty:
    st.warning(f"No data for {symbol}")
    st.stop()

# === Refresh buttons ===
st.sidebar.header("🔄 Data Maintenance")
colr1, colr2 = st.sidebar.columns(2)
if colr1.button("Refresh 6mo"):
    new = fetch_live_and_save(symbol, period="6mo")
    st.sidebar.success(f"Refreshed {symbol} (6mo): {len(new)} rows" if not new.empty else "No new data")
if colr2.button("Backfill 1y"):
    new = fetch_live_and_save(symbol, period="1y")
    st.sidebar.success(f"Backfilled {symbol} (1y): {len(new)} rows" if not new.empty else "No new data")

# === Filter by date ===
df["Date"] = pd.to_datetime(df["Date"])
df = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))]
if df.empty:
    st.warning("No data for selected date range.")
    st.stop()

# === Indicators for main panel ===
df["RSI"] = RSIIndicator(close=df["Close"]).rsi()
df["SMA20"] = SMAIndicator(close=df["Close"], window=20).sma_indicator()

# === Crossover Alert (visual) ===
last_close = df["Close"].iloc[-1]
last_sma = df["SMA20"].iloc[-1]
if last_close > last_sma:
    st.success("📈 Potential Buy Signal: Price crossed above SMA20")
elif last_close < last_sma:
    st.error("📉 Potential Sell Signal: Price crossed below SMA20")
else:
    st.info("⏳ No crossover detected.")

# === Price Chart ===
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="OHLC"
))
fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA20"], line=dict(color="blue", width=2), name="SMA20"))
fig.update_layout(title=f"{symbol} Price Chart", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig, use_container_width=True)

# === RSI Chart ===
rsi_fig = go.Figure()
rsi_fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], line=dict(color="green"), name="RSI"))
rsi_fig.update_layout(title="Relative Strength Index", yaxis=dict(range=[0, 100]))
st.plotly_chart(rsi_fig, use_container_width=True)

# === Alerts & Rules (Sidebar) ===
st.sidebar.header("🚨 Alerts & Rules")
rules = load_rules(symbol)
rules["rsi_low"] = st.sidebar.number_input("RSI Low (oversold)", min_value=0.0, max_value=100.0, value=float(rules["rsi_low"]), step=1.0)
rules["rsi_high"] = st.sidebar.number_input("RSI High (overbought)", min_value=0.0, max_value=100.0, value=float(rules["rsi_high"]), step=1.0)
rules["move_1d_abs_gt"] = st.sidebar.number_input("1D Move ≥ %", min_value=0.0, max_value=50.0, value=float(rules["move_1d_abs_gt"]), step=0.5)
rules["gap_abs_gt"] = st.sidebar.number_input("Gap Open ≥ %", min_value=0.0, max_value=50.0, value=float(rules["gap_abs_gt"]), step=0.5)
rules["near_52w_high_within"] = st.sidebar.number_input("Within % of 52w High", min_value=0.0, max_value=20.0, value=float(rules["near_52w_high_within"]), step=0.5)
rules["near_52w_low_within"]  = st.sidebar.number_input("Within % of 52w Low",  min_value=0.0, max_value=20.0, value=float(rules["near_52w_low_within"]),  step=0.5)

if st.sidebar.button("💾 Save Rules"):
    save_rules(symbol, rules)
    st.sidebar.success("Rules saved")

# === Alerts panel for selected symbol ===
st.subheader(f"🚨 Alerts for {symbol}")
signals = compute_signals_df(df)
active_alerts = apply_alerts(signals, rules)
if active_alerts:
    for a in active_alerts:
        st.warning(f"• {a}")
else:
    st.info("No alerts triggered for current rules.")

# === Correlation Matrix ===
st.header("📊 Correlation Matrix")
selected_corr_symbols = st.multiselect("Select symbols to compare", default_symbols, default=default_symbols[:4])
if selected_corr_symbols:
    corr_df = pd.DataFrame()
    for sym in selected_corr_symbols:
        try:
            data = get_or_fetch(sym)
            if not data.empty:
                s = data[["Date","Close"]].set_index("Date")["Close"].rename(sym)
                corr_df = pd.concat([corr_df, s], axis=1)
        except Exception as e:
            st.warning(f"Couldn't load {sym}: {e}")
    if not corr_df.empty:
        corr_matrix = corr_df.corr()
        fig_corr, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig_corr)

# === Analytical Report (reuse your earlier section if present) ===
st.header("📊 Analytical Report — Latest Changes")
watchlist = st.multiselect("Choose symbols to analyze", options=default_symbols, default=default_symbols[:10])
extra = st.text_input("Add another symbol to this report (optional):", "")
if extra.strip():
    watchlist = list(dict.fromkeys(watchlist + [extra.upper()]))

def compute_snapshot(df: pd.DataFrame, sym: str) -> dict:
    df = df.sort_values("Date").reset_index(drop=True)
    close = df["Close"].astype(float)
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)
    def pct(n):
        if len(close) > n:
            return (close.iloc[-1] / close.iloc[-1-n] - 1.0) * 100.0
        return None
    r_1d  = pct(1); r_1w  = pct(5); r_1m  = pct(21); r_3m  = pct(63)
    rsi = RSIIndicator(close=close).rsi().iloc[-1] if len(close) > 14 else None
    sma20 = SMAIndicator(close=close, window=20).sma_indicator().iloc[-1] if len(close) >= 20 else None
    sma50 = SMAIndicator(close=close, window=50).sma_indicator().iloc[-1] if len(close) >= 50 else None
    trend = None
    if sma20 is not None and sma50 is not None:
        trend = "SMA20 > SMA50 (uptrend)" if sma20 > sma50 else ("SMA20 < SMA50 (downtrend)" if sma20 < sma50 else "SMA20 = SMA50 (flat)")
    vol = None
    if len(close) >= 21:
        ret = close.pct_change()
        vol = (ret.rolling(20).std().iloc[-1] * (252**0.5)) * 100.0
    lookback = min(len(high), 252)
    near_high = near_low = None
    if lookback >= 1:
        h52 = high.iloc[-lookback:].max()
        l52 = low.iloc[-lookback:].min()
        last = close.iloc[-1]
        near_high = (last / h52 - 1.0) * 100.0 if h52 else None
        near_low  = (last / l52 - 1.0) * 100.0 if l52 else None
    gap = (df["Open"].iloc[-1] / df["Close"].iloc[-2] - 1.0) * 100.0 if len(df) >= 2 else None
    return {
        "Symbol": sym.upper(),
        "Last Close": round(close.iloc[-1], 4),
        "1D %": None if r_1d is None else round(r_1d, 2),
        "1W %": None if r_1w is None else round(r_1w, 2),
        "1M %": None if r_1m is None else round(r_1m, 2),
        "3M %": None if r_3m is None else round(r_3m, 2),
        "RSI": None if rsi is None else round(rsi, 1),
        "SMA20": None if sma20 is None else round(sma20, 4),
        "SMA50": None if sma50 is None else round(sma50, 4),
        "Trend": trend,
        "Vol (ann % ~20d)": None if vol is None else round(vol, 1),
        "Δ vs 52W High %": None if near_high is None else round(near_high, 2),
        "Δ vs 52W Low %":  None if near_low  is None else round(near_low,  2),
        "Gap Today %": None if gap is None else round(gap, 2),
    }

rows = []
if watchlist:
    with st.spinner("Building report..."):
        for sym in watchlist:
            data = get_or_fetch(sym)
            if data is None or data.empty:
                st.warning(f"No data for {sym}")
                continue
            snap = compute_snapshot(data, sym)
            rows.append(snap)

if rows:
    report_df = pd.DataFrame(rows)
    if "1D %" in report_df.columns and report_df["1D %"].notna().any():
        report_df = report_df.sort_values(by="1D %", ascending=False, na_position="last")
    else:
        report_df = report_df.sort_values(by="Symbol")
    st.dataframe(report_df, use_container_width=True)

    # CSV download for report
    csv_bytes = report_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download report as CSV",
        data=csv_bytes,
        file_name=f"quantflow_report_{pd.Timestamp.now().date()}.csv",
        mime="text/csv",
    )

    # Top movers (optional)
    movers = report_df.dropna(subset=["1D %"]).nlargest(10, "1D %")[["Symbol","1D %"]]
    if not movers.empty:
        st.subheader("Top 10 Movers (1D)")
        bar = go.Figure(data=[go.Bar(x=movers["Symbol"], y=movers["1D %"])])
        bar.update_layout(yaxis_title="1D %", xaxis_title="Symbol")
        st.plotly_chart(bar, use_container_width=True)

    # Refresh selected watchlist
    if st.button("🔄 Refresh watchlist (6mo)"):
        total = 0
        for sym in watchlist:
            new = fetch_live_and_save(sym, period="6mo")
            total += (0 if new is None else len(new))
        st.success(f"Refreshed {len(watchlist)} symbols. Inserted ~{total} rows (if available).")

import requests, os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")  # set to your Render URL in prod

st.sidebar.header("📣 Notifications")
tg_syms = st.sidebar.text_input("Symbols to scan (comma-separated)", ",".join(default_symbols[:10]))
tg_top = st.sidebar.number_input("Top N (by |1D|)", min_value=0, value=10, step=1)
tg_dry = st.sidebar.checkbox("Dry run (don't send)", value=True)

if st.sidebar.button("Send Alerts to Telegram"):
    try:
        url = f"{API_URL}/notify/telegram"
        params = dict(symbols=tg_syms, top=int(tg_top))
        payload = {"dry_run": bool(tg_dry)}
        r = requests.post(url, params=params, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        if data.get("sent"):
            st.sidebar.success("Alerts sent to Telegram ✅")
        else:
            st.sidebar.info(data.get("message", "Dry run complete."))
    except Exception as e:
        st.sidebar.error(f"Failed: {e}")


from common.telegram import send_telegram_message

def check_alerts_and_notify(symbol, df):
    last_close = df["Close"].iloc[-1]
    sma20 = df["SMA20"].iloc[-1]
    rsi = df["RSI"].iloc[-1]

    if last_close > sma20:
        send_telegram_message(f"📈 {symbol} crossed above SMA20! Close={last_close:.2f}")
    elif last_close < sma20:
        send_telegram_message(f"📉 {symbol} crossed below SMA20! Close={last_close:.2f}")

    if rsi > 70:
        send_telegram_message(f"⚠️ {symbol} RSI={rsi:.1f} (Overbought)")
    elif rsi < 30:
        send_telegram_message(f"⚠️ {symbol} RSI={rsi:.1f} (Oversold)")
