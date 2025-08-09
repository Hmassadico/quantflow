import os
import yfinance as yf
import pandas as pd
from sqlalchemy import text
from common.db import get_engine, ensure_schema

def fetch_stock_data(symbol: str, period="6mo", interval="1d"):
    df = yf.download(symbol, period=period, interval=interval)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    if "Adj Close" in df.columns:
        df.rename(columns={"Adj Close": "Adj_Close"}, inplace=True)
    else:
        df["Adj_Close"] = df["Close"]
    df["Symbol"] = symbol.upper()
    return df[["Date","Open","High","Low","Close","Adj_Close","Volume","Symbol"]]

def save_to_db(df: pd.DataFrame):
    engine = get_engine()
    ensure_schema(engine)
    # Upsert-light: append; for real upsert, use ON CONFLICT (needs key)
    df.to_sql("stocks", con=engine, if_exists="append", index=False, method="multi", chunksize=1000)

if __name__ == "__main__":
    sym = os.getenv("SYMBOL","AAPL")
    df = fetch_stock_data(sym)
    save_to_db(df)
    print(f"✅ Saved {len(df)} rows for {sym}")
