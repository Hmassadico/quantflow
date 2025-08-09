# common/db.py
import os
from pathlib import Path
from sqlalchemy import create_engine, text

# Load .env if present (pip install python-dotenv)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

def get_engine():
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        # Supabase Postgres
        return create_engine(db_url, pool_pre_ping=True)

    # Fallback: local DuckDB (needs `duckdb-engine` in requirements)
    duck_path = Path(__file__).resolve().parents[1] / "data" / "market_data.duckdb"
    duck_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"duckdb:///{duck_path}")

DDL = """
CREATE TABLE IF NOT EXISTS stocks (
  "Date" TIMESTAMP,
  "Open" DOUBLE PRECISION,
  "High" DOUBLE PRECISION,
  "Low"  DOUBLE PRECISION,
  "Close" DOUBLE PRECISION,
  "Adj_Close" DOUBLE PRECISION,
  "Volume" BIGINT,
  "Symbol" TEXT
);
CREATE INDEX IF NOT EXISTS idx_stocks_symbol_date ON stocks("Symbol","Date");
"""

def ensure_schema(engine):
    with engine.begin() as conn:
        conn.execute(text(DDL))
