# init_db.py
import duckdb
from pathlib import Path

DB_PATH = Path("data/market_data.duckdb")
TABLE_NAME = "stocks"

# Create parent dir if needed
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Create DB and table if not exists
con = duckdb.connect(str(DB_PATH))
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
con.close()

print("✅ DuckDB initialized.")
