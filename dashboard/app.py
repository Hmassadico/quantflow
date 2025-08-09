import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import text
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from common.db import get_engine, ensure_schema

# === Setup ===
st.set_page_config(page_title="QuantFlow", layout="wide")
st.title("📈 QuantFlow Dashboard")

engine = get_engine()
ensure_schema(engine)
TABLE_NAME = "stocks"

default_symbols = ["AAPL", "MSFT", "GOOG", "TSLA", "NVDA", "META", "AMZN"]
symbol = st.selectbox("📌 Select a stock", default_symbols, index=0)
custom_symbol = st.text_input("Or enter another symbol:", "")

if custom_symbol.strip():
    symbol = custom_symbol.upper()

# === Date Filter ===
st.sidebar.header("📅 Date Range")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2024-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

def load_from_db(sym: str) -> pd.DataFrame:
    q = text('SELECT * FROM stocks WHERE "Symbol" = :s ORDER BY "Date"')
    return pd.read_sql(q, engine, params={"s": sym.upper()})

def fetch_live_and_save(sym: str) -> pd.DataFrame:
    df = yf.download(sym, period="6mo", interval="1d")
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
    df = df[["Date", "Open", "High", "Low", "Close", "Adj_Close", "Volume", "Symbol"]]
    # append
    df.to_sql(TABLE_NAME, con=engine, if_exists="append", index=False, method="multi", chunksize=1000)
    return df

# === Get Stock Data from DB or yFinance ===
df = load_from_db(symbol)

if df.empty:
    st.warning(f"🔄 No data for {symbol}. Fetching live...")
    df = fetch_live_and_save(symbol)
    if df.empty:
        st.error("Symbol not found.")
        st.stop()
    st.success(f"✅ Data for {symbol} saved!")

# === Filter by date ===
df["Date"] = pd.to_datetime(df["Date"])
df = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))]

if df.empty:
    st.warning("No data for selected date range.")
    st.stop()

# === Indicators ===
df["RSI"] = RSIIndicator(close=df["Close"]).rsi()
df["SMA20"] = SMAIndicator(close=df["Close"], window=20).sma_indicator()

# === Crossover Alert ===
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
    x=df["Date"],
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    name="OHLC"
))
fig.add_trace(go.Scatter(
    x=df["Date"], y=df["SMA20"],
    line=dict(color="blue", width=2),
    name="SMA20"
))
fig.update_layout(title=f"{symbol} Price Chart", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig, use_container_width=True)

# === RSI Chart ===
rsi_fig = go.Figure()
rsi_fig.add_trace(go.Scatter(
    x=df["Date"], y=df["RSI"],
    line=dict(color="green"),
    name="RSI"
))
rsi_fig.update_layout(title="Relative Strength Index", yaxis=dict(range=[0, 100]))
st.plotly_chart(rsi_fig, use_container_width=True)

# === Correlation Matrix ===
st.header("📊 Correlation Matrix")
selected_corr_symbols = st.multiselect("Select symbols to compare", default_symbols, default=default_symbols[:4])

if selected_corr_symbols:
    corr_df = pd.DataFrame()
    for sym in selected_corr_symbols:
        try:
            q = text('SELECT "Date","Close" FROM stocks WHERE "Symbol" = :s ORDER BY "Date"')
            data = pd.read_sql(q, engine, params={"s": sym.upper()})
            if not data.empty:
                data = data.set_index("Date")["Close"].rename(sym)
                corr_df = pd.concat([corr_df, data], axis=1)
        except Exception as e:
            st.warning(f"Couldn't load {sym}: {e}")

    if not corr_df.empty:
        corr_matrix = corr_df.corr()
        fig_corr, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig_corr)
