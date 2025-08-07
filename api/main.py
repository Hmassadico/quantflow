from fastapi import FastAPI, HTTPException, Query, Request
from typing import Optional
import duckdb
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from pathlib import Path
from fastapi.responses import JSONResponse

app = FastAPI(title="QuantFlow API")

DB_PATH = Path("../data/market_data.duckdb")
TABLE_NAME = "stocks"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
con = duckdb.connect(str(DB_PATH))

def fetch_and_save(symbol: str):
    df = yf.download(symbol, period="6mo", interval="1d")

    if df.empty:
        return None

    df.reset_index(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Adj Close" in df.columns:
        df.rename(columns={"Adj Close": "Adj_Close"}, inplace=True)
    else:
        df["Adj_Close"] = df["Close"]

    df["Symbol"] = symbol.upper()
    df = df[["Date", "Open", "High", "Low", "Close", "Adj_Close", "Volume", "Symbol"]]
    
    con = duckdb.connect(str(DB_PATH))
    con.register("df", df)
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            Date TIMESTAMP,
            Open DOUBLE,
            High DOUBLE,
            Low DOUBLE,
            Close DOUBLE,
            Adj_Close DOUBLE,
            Volume BIGINT,
            Symbol TEXT
        );
    """)
    con.execute(f"""
        INSERT INTO {TABLE_NAME}
        SELECT Date, Open, High, Low, Close, Adj_Close, Volume, Symbol FROM df
    """)
    return df

def load_stock_data(symbol: str) -> Optional[pd.DataFrame]:
    with duckdb.connect(str(DB_PATH)) as con:
        query = f"SELECT * FROM {TABLE_NAME} WHERE Symbol = '{symbol.upper()}' ORDER BY Date"
        df = con.execute(query).fetchdf()
        print(df)
        if df.empty:
            df = fetch_and_save(symbol)
        return df





def filter_by_date(df: pd.DataFrame, start: Optional[str], end: Optional[str]):
    df["Date"] = pd.to_datetime(df["Date"])
    if start:
        df = df[df["Date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["Date"] <= pd.to_datetime(end)]
    return df

@app.get("/")
def root():
    return {"message": "Welcome to QuantFlow API"}

@app.get("/stocks/{symbol}")
def get_stock(symbol: str, start: Optional[str] = Query(None), end: Optional[str] = Query(None)):
    df = load_stock_data(symbol)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="Stock data not found.")
    df = filter_by_date(df, start, end)
    return df.to_dict(orient="records")

@app.get("/indicators/{symbol}")
def get_indicators(symbol: str, start: Optional[str] = Query(None), end: Optional[str] = Query(None)):
    df = load_stock_data(symbol)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="Stock data not found.")
    df = filter_by_date(df, start, end)

    # Calculate indicators
    df["RSI"] = RSIIndicator(close=df["Close"]).rsi()
    df["SMA20"] = SMAIndicator(close=df["Close"], window=20).sma_indicator()

    return df[["Date", "Close", "RSI", "SMA20"]].dropna().to_dict(orient="records")


@app.get("/correlation")
def get_correlation(
    request: Request,
    symbols: str = Query(..., description="Comma-separated stock symbols (e.g. AAPL,MSFT,NVDA)"),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None)
):
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    
    if len(symbol_list) < 2:
        raise HTTPException(status_code=400, detail="Please provide at least 2 symbols.")

    df_list = []

    
    try:
        for sym in symbol_list:
            df = load_stock_data(sym)  # ✅ pass it in
            if df is None or df.empty:
                continue

            df = filter_by_date(df, start, end)
            df = df[["Date", "Close"]].copy()
            df = df.rename(columns={"Close": sym})
            df = df.set_index("Date")
            df_list.append(df)
    finally:
        con.close()  # ✅ safely close

    if not df_list:
        raise HTTPException(status_code=404, detail="No valid data found for selected symbols.")

    combined = pd.concat(df_list, axis=1).dropna()

    if combined.empty:
        raise HTTPException(status_code=404, detail="Not enough overlapping data for correlation.")

    corr = combined.corr().round(4)
    return JSONResponse(content=corr.to_dict())
@app.on_event("shutdown")
def close_connection():
    con.close()
