from fastapi import FastAPI, HTTPException, Query
from typing import Optional, List
import pandas as pd
import yfinance as yf
from sqlalchemy import text
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from common.db import get_engine, ensure_schema

app = FastAPI(title="QuantFlow API")
engine = get_engine()
ensure_schema(engine)

def fetch_and_save(symbol: str):
    df = yf.download(symbol, period="6mo", interval="1d")
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    if "Adj Close" in df.columns:
        df.rename(columns={"Adj Close": "Adj_Close"}, inplace=True)
    else:
        df["Adj_Close"] = df["Close"]
    df["Symbol"] = symbol.upper()
    df = df[["Date","Open","High","Low","Close","Adj_Close","Volume","Symbol"]]
    df.to_sql("stocks", con=engine, if_exists="append", index=False, method="multi", chunksize=1000)
    return df

def load_stock_data(symbol: str) -> pd.DataFrame:
    q = text('SELECT * FROM stocks WHERE "Symbol" = :s ORDER BY "Date"')
    df = pd.read_sql(q, engine, params={"s": symbol.upper()})
    if df.empty:
        df = fetch_and_save(symbol)
    return df

def filter_by_date(df: pd.DataFrame, start: Optional[str], end: Optional[str]):
    df["Date"] = pd.to_datetime(df["Date"])
    if start: df = df[df["Date"] >= pd.to_datetime(start)]
    if end:   df = df[df["Date"] <= pd.to_datetime(end)]
    return df

@app.get("/")
def root():
    return {"message": "Welcome to QuantFlow API"}

@app.get("/stocks/{symbol}")
def get_stock(symbol: str, start: Optional[str] = None, end: Optional[str] = None):
    df = load_stock_data(symbol)
    if df is None or df.empty:
        raise HTTPException(404, "Stock data not found.")
    df = filter_by_date(df, start, end)
    return df.to_dict(orient="records")

@app.get("/indicators/{symbol}")
def get_indicators(symbol: str, start: Optional[str] = None, end: Optional[str] = None):
    df = load_stock_data(symbol)
    if df is None or df.empty:
        raise HTTPException(404, "Stock data not found.")
    df = filter_by_date(df, start, end)
    df["RSI"] = RSIIndicator(close=df["Close"]).rsi()
    df["SMA20"] = SMAIndicator(close=df["Close"], window=20).sma_indicator()
    return df[["Date","Close","RSI","SMA20"]].dropna().to_dict(orient="records")

@app.get("/correlation")
def get_correlation(symbols: str, start: Optional[str] = None, end: Optional[str] = None):
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if len(symbol_list) < 2:
        raise HTTPException(400, "Please provide at least 2 symbols.")
    series = []
    for sym in symbol_list:
        df = load_stock_data(sym)
        if df is None or df.empty: 
            continue
        df = filter_by_date(df, start, end)
        s = df[["Date","Close"]].rename(columns={"Close": sym}).set_index("Date")[sym]
        series.append(s)
    if not series:
        raise HTTPException(404, "No valid data.")
    combined = pd.concat(series, axis=1).dropna()
    if combined.empty:
        raise HTTPException(404, "Not enough overlap.")
    return combined.corr().round(4).to_dict()
