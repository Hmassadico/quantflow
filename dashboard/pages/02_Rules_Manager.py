# dashboard/pages/02_Rules_Manager.py
import streamlit as st
import pandas as pd
from sqlalchemy import text
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.db import get_engine

st.set_page_config(page_title="QuantFlow – Rules Manager", layout="wide")
st.title("⚙️ Rules Manager")

engine = get_engine()

# Ensure table exists (in case this page is hit first)
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

def load_rules_df():
    q = text('SELECT * FROM alert_rules ORDER BY "Symbol"')
    df = pd.read_sql(q, engine)
    if df.empty:
        df = pd.DataFrame(columns=[
            "Symbol","rsi_low","rsi_high","move_1d_abs_gt","gap_abs_gt","near_52w_high_within","near_52w_low_within","updated_at"
        ])
    return df

def save_bulk(df: pd.DataFrame):
    # Upsert each row
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
            for _, r in df.iterrows():
                conn.execute(up, dict(
                    s=r["Symbol"].upper(),
                    rsi_low=float(r["rsi_low"]),
                    rsi_high=float(r["rsi_high"]),
                    m1d=float(r["move_1d_abs_gt"]),
                    gap=float(r["gap_abs_gt"]),
                    nh=float(r["near_52w_high_within"]),
                    nl=float(r["near_52w_low_within"]),
                ))
    else:
        with engine.begin() as conn:
            for _, r in df.iterrows():
                conn.execute(text('DELETE FROM alert_rules WHERE "Symbol" = :s'), {"s": r["Symbol"].upper()})
                conn.execute(text("""
                    INSERT INTO alert_rules ("Symbol","rsi_low","rsi_high","move_1d_abs_gt","gap_abs_gt","near_52w_high_within","near_52w_low_within","updated_at")
                    VALUES (:s,:rsi_low,:rsi_high,:m1d,:gap,:nh,:nl, CURRENT_TIMESTAMP)
                """), dict(
                    s=r["Symbol"].upper(),
                    rsi_low=float(r["rsi_low"]),
                    rsi_high=float(r["rsi_high"]),
                    m1d=float(r["move_1d_abs_gt"]),
                    gap=float(r["gap_abs_gt"]),
                    nh=float(r["near_52w_high_within"]),
                    nl=float(r["near_52w_low_within"]),
                ))

# Load existing
rules_df = load_rules_df()

st.info("Tip: You can paste new rows into the table below. Symbol must be unique.")

edited = st.data_editor(
    rules_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Symbol": st.column_config.TextColumn(help="Ticker symbol, e.g., AAPL"),
        "rsi_low": st.column_config.NumberColumn(min_value=0, max_value=100, step=1),
        "rsi_high": st.column_config.NumberColumn(min_value=0, max_value=100, step=1),
        "move_1d_abs_gt": st.column_config.NumberColumn(min_value=0, step=0.1),
        "gap_abs_gt": st.column_config.NumberColumn(min_value=0, step=0.1),
        "near_52w_high_within": st.column_config.NumberColumn(min_value=0, step=0.1),
        "near_52w_low_within": st.column_config.NumberColumn(min_value=0, step=0.1),
    },
    key="rules_table",
)

c1, c2, c3 = st.columns(3)
if c1.button("💾 Save All Changes"):
    # sanitize
    if "Symbol" in edited.columns:
        edited["Symbol"] = edited["Symbol"].fillna("").str.upper().str.strip()
        edited = edited[edited["Symbol"] != ""]
    save_bulk(edited)
    st.success("Saved rules")

if c2.button("➕ Add Default Row"):
    new_row = pd.DataFrame([{
        "Symbol": "AAPL",
        "rsi_low": 30, "rsi_high": 70,
        "move_1d_abs_gt": 3.0, "gap_abs_gt": 2.0,
        "near_52w_high_within": 1.0, "near_52w_low_within": 1.0,
    }])
    rules_df = pd.concat([rules_df, new_row], ignore_index=True)
    st.session_state["rules_table"] = rules_df  # refresh UI
    st.rerun()

if c3.button("🔄 Reload"):
    st.rerun()
