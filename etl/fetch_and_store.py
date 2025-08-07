import yfinance as yf
import pandas as pd
import duckdb
from pathlib import Path

# Define database path
DB_PATH = Path("data/market_data.duckdb")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)  # Ensure /data folder exists

def fetch_stock_data(symbol: str, period="6mo", interval="1d"):
    df = yf.download(symbol, period=period, interval=interval)

    # Flatten columns if MultiIndex (e.g., from yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)

    # If 'Adj Close' exists, rename it
    if "Adj Close" in df.columns:
        df.rename(columns={"Adj Close": "Adj_Close"}, inplace=True)
    else:
        df["Adj_Close"] = df["Close"]  # fallback to Close if not available

    # Rename others for consistency
    df.rename(columns={
        "Date": "Date",
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Volume": "Volume"
    }, inplace=True)

    df["Symbol"] = symbol

    # Final column order
    expected_cols = ["Date", "Open", "High", "Low", "Close", "Adj_Close", "Volume", "Symbol"]
    df = df[[col for col in expected_cols if col in df.columns]]

    return df






def save_to_duckdb(df: pd.DataFrame, table_name="stocks"):
    con = duckdb.connect(str(DB_PATH))

    # Create table if it doesn't exist
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
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

    # Register and insert data
    con.register("df", df)
    con.execute(f"""
        INSERT INTO {table_name}
        SELECT Date, Open, High, Low, Close, Adj_Close, Volume, Symbol FROM df
    """)

    con.close()

if __name__ == "__main__":
    symbol = "AAPL"
    df = fetch_stock_data(symbol)
    print("✅ Preview of downloaded data:")
    print(df.head())
    save_to_duckdb(df)
    print(f"✅ Saved {len(df)} rows of {symbol} to {DB_PATH}")
