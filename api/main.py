from fastapi import FastAPI, HTTPException, Query
from typing import Optional, List
import pandas as pd
import yfinance as yf
from sqlalchemy import text
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from pathlib import Path
from pydantic import BaseModel, Field
from fastapi import Body


import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.db import get_engine, ensure_schema
from common.telegram import send_telegram_message


app = FastAPI(title="QuantFlow API")
engine = get_engine()
ensure_schema(engine)


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


class RuleIn(BaseModel):
    rsi_low: float = Field(default=30.0, ge=0, le=100)
    rsi_high: float = Field(default=70.0, ge=0, le=100)
    move_1d_abs_gt: float = Field(default=3.0, ge=0)
    gap_abs_gt: float = Field(default=2.0, ge=0)
    near_52w_high_within: float = Field(default=1.0, ge=0)
    near_52w_low_within: float = Field(default=1.0, ge=0)

class RuleOut(RuleIn):
    Symbol: str
    updated_at: Optional[str] = None

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


def compute_snapshot_df(df: pd.DataFrame, sym: str) -> dict:
    df = df.sort_values("Date").reset_index(drop=True)
    close = df["Close"].astype(float)
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)

    def pct(n):
        if len(close) > n:
            return (close.iloc[-1] / close.iloc[-1-n] - 1.0) * 100.0
        return None

    r_1d  = pct(1)   # ~1 day
    r_1w  = pct(5)   # ~1 week
    r_1m  = pct(21)  # ~1 month
    r_3m  = pct(63)  # ~3 months

    rsi = RSIIndicator(close=close).rsi().iloc[-1] if len(close) > 14 else None
    sma20 = SMAIndicator(close=close, window=20).sma_indicator().iloc[-1] if len(close) >= 20 else None
    sma50 = SMAIndicator(close=close, window=50).sma_indicator().iloc[-1] if len(close) >= 50 else None

    trend = None
    if sma20 is not None and sma50 is not None:
        if sma20 > sma50: trend = "SMA20 > SMA50 (uptrend)"
        elif sma20 < sma50: trend = "SMA20 < SMA50 (downtrend)"
        else: trend = "SMA20 = SMA50 (flat)"

    vol = None
    if len(close) >= 21:
        ret = close.pct_change()
        vol = (ret.rolling(20).std().iloc[-1] * (252**0.5)) * 100.0  # annualized %

    lookback = min(len(high), 252)
    near_high = near_low = None
    if lookback >= 1:
        h52 = high.iloc[-lookback:].max()
        l52 = low.iloc[-lookback:].min()
        last = close.iloc[-1]
        near_high = (last / h52 - 1.0) * 100.0 if h52 else None
        near_low  = (last / l52 - 1.0) * 100.0 if l52 else None

    gap = None
    if len(df) >= 2:
        gap = (df["Open"].iloc[-1] / df["Close"].iloc[-2] - 1.0) * 100.0

    return {
        "Symbol": sym.upper(),
        "LastClose": round(close.iloc[-1], 4),
        "Pct_1D": None if r_1d is None else round(r_1d, 2),
        "Pct_1W": None if r_1w is None else round(r_1w, 2),
        "Pct_1M": None if r_1m is None else round(r_1m, 2),
        "Pct_3M": None if r_3m is None else round(r_3m, 2),
        "RSI": None if rsi is None else round(rsi, 1),
        "SMA20": None if sma20 is None else round(sma20, 4),
        "SMA50": None if sma50 is None else round(sma50, 4),
        "Trend": trend,
        "VolAnnPct_20d": None if vol is None else round(vol, 1),
        "Delta_vs_52W_High_pct": None if near_high is None else round(near_high, 2),
        "Delta_vs_52W_Low_pct":  None if near_low  is None else round(near_low,  2),
        "GapTodayPct": None if gap is None else round(gap, 2),
    }

import numpy as np

def _rolling_extrema(close: pd.Series, high: pd.Series, low: pd.Series, lb: int = 252):
    # 52-week high/low using last ~252 trading days (if available)
    look = min(len(close), lb)
    if look < 1:
        return None, None
    h52 = high.iloc[-look:].max()
    l52 = low.iloc[-look:].min()
    return h52, l52

def compute_signals_df(df: pd.DataFrame) -> dict:
    """
    Compute point-in-time signals for the latest bar:
    - rsi latest + crosses of 30/70
    - sma20, sma50; golden/death cross (today vs yesterday)
    - 52w breakout
    - 1D return threshold
    - Gap open threshold
    """
    df = df.sort_values("Date").reset_index(drop=True)
    close = df["Close"].astype(float)
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)
    openp = df["Open"].astype(float)

    out = {}

    # RSI
    rsi_series = RSIIndicator(close=close).rsi() if len(close) >= 15 else pd.Series([np.nan]*len(close))
    rsi_now = rsi_series.iloc[-1]
    rsi_prev = rsi_series.iloc[-2] if len(rsi_series) >= 2 else np.nan
    out["rsi"] = None if np.isnan(rsi_now) else round(float(rsi_now), 2)
    out["rsi_cross_70_up"]   = bool(rsi_prev < 70 <= rsi_now) if not np.isnan(rsi_prev) else False
    out["rsi_cross_70_down"] = bool(rsi_prev > 70 >= rsi_now) if not np.isnan(rsi_prev) else False
    out["rsi_cross_30_up"]   = bool(rsi_prev < 30 <= rsi_now) if not np.isnan(rsi_prev) else False
    out["rsi_cross_30_down"] = bool(rsi_prev > 30 >= rsi_now) if not np.isnan(rsi_prev) else False

    # SMAs + crosses
    sma20_series = SMAIndicator(close=close, window=20).sma_indicator() if len(close) >= 20 else pd.Series([np.nan]*len(close))
    sma50_series = SMAIndicator(close=close, window=50).sma_indicator() if len(close) >= 50 else pd.Series([np.nan]*len(close))
    s20_now = sma20_series.iloc[-1]
    s50_now = sma50_series.iloc[-1]
    s20_prev = sma20_series.iloc[-2] if len(sma20_series) >= 2 else np.nan
    s50_prev = sma50_series.iloc[-2] if len(sma50_series) >= 2 else np.nan

    out["sma20"] = None if np.isnan(s20_now) else round(float(s20_now), 4)
    out["sma50"] = None if np.isnan(s50_now) else round(float(s50_now), 4)
    out["golden_cross_today"] = (s20_prev < s50_prev) and (s20_now >= s50_now) if not any(np.isnan([s20_prev,s50_prev,s20_now,s50_now])) else False
    out["death_cross_today"]  = (s20_prev > s50_prev) and (s20_now <= s50_now) if not any(np.isnan([s20_prev,s50_prev,s20_now,s50_now])) else False

    # 52-week breakout
    h52, l52 = _rolling_extrema(close, high, low, lb=252)
    last = close.iloc[-1]
    out["breakout_52w_high"] = (h52 is not None) and (last >= h52)
    out["breakdown_52w_low"] = (l52 is not None) and (last <= l52)
    out["pct_to_52w_high"] = None if not h52 else round((last / h52 - 1.0) * 100.0, 2)
    out["pct_to_52w_low"]  = None if not l52 else round((last / l52 - 1.0) * 100.0, 2)

    # 1D return and gap (%)
    ret_1d = np.nan
    if len(close) >= 2:
        ret_1d = (last / close.iloc[-2] - 1.0) * 100.0
    out["ret_1d_pct"] = None if np.isnan(ret_1d) else round(float(ret_1d), 2)

    gap_pct = np.nan
    if len(openp) >= 2 and len(close) >= 2:
        gap_pct = (openp.iloc[-1] / close.iloc[-2] - 1.0) * 100.0
    out["gap_open_pct"] = None if np.isnan(gap_pct) else round(float(gap_pct), 2)

    return out

def apply_alert_rules(signals: dict,
                      rsi_low: float = 30, rsi_high: float = 70,
                      move_1d_abs_gt: float = 3.0,
                      gap_abs_gt: float = 2.0,
                      near_52w_high_within: float = 1.0,
                      near_52w_low_within: float = 1.0) -> List[str]:
    """
    Return a list of human-readable alerts based on thresholds.
    """
    alerts = []

    rsi = signals.get("rsi")
    if rsi is not None:
        if rsi <= rsi_low:
            alerts.append(f"RSI ≤ {rsi_low} (oversold) [{rsi}]")
        if rsi >= rsi_high:
            alerts.append(f"RSI ≥ {rsi_high} (overbought) [{rsi}]")
    if signals.get("rsi_cross_30_up"):   alerts.append("RSI crossed up through 30 (bullish)")
    if signals.get("rsi_cross_70_down"): alerts.append("RSI crossed down through 70 (bearish)")

    if signals.get("golden_cross_today"): alerts.append("Golden cross today (SMA20 ↑ over SMA50)")
    if signals.get("death_cross_today"):  alerts.append("Death cross today (SMA20 ↓ under SMA50)")

    # 1D move
    m1d = signals.get("ret_1d_pct")
    if m1d is not None and abs(m1d) >= move_1d_abs_gt:
        alerts.append(f"1D move ≥ {move_1d_abs_gt}% [{round(m1d,2)}%]")

    # Gap
    gap = signals.get("gap_open_pct")
    if gap is not None and abs(gap) >= gap_abs_gt:
        alerts.append(f"Gap open ≥ {gap_abs_gt}% [{round(gap,2)}%]")

    # Near 52w
    p2h = signals.get("pct_to_52w_high")
    if p2h is not None and p2h >= -near_52w_high_within and p2h <= 0:
        alerts.append(f"Within {near_52w_high_within}% of 52w high [{p2h}%]")
    p2l = signals.get("pct_to_52w_low")
    if p2l is not None and p2l <= near_52w_low_within and p2l >= 0:
        alerts.append(f"Within {near_52w_low_within}% of 52w low [{p2l}%]")

    # Explicit breakouts/breakdowns
    if signals.get("breakout_52w_high"): alerts.append("New 52w high breakout")
    if signals.get("breakdown_52w_low"): alerts.append("New 52w low breakdown")

    return alerts



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


@app.get("/report/watchlist")
def report_watchlist(
    symbols: str = Query(..., description="Comma-separated symbols, e.g., AAPL,MSFT,NVDA"),
    start: Optional[str] = None,
    end: Optional[str] = None,
    top: Optional[int] = Query(0, description="Return top N by 1D % (0 = all)")
):
    """
    Returns an analytical report for a watchlist, similar to the dashboard table.
    - symbols: comma-separated list
    - start/end: optional YYYY-MM-DD filters
    - top: if >0, return only top N by 1D % change
    """
    syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not syms:
        raise HTTPException(400, "No symbols provided.")

    rows = []
    for sym in syms:
        df = load_stock_data(sym)
        if df is None or df.empty:
            # try to fetch live data once
            fetched = fetch_and_save(sym)
            df = fetched if fetched is not None else pd.DataFrame()

        if df.empty:
            continue

        df = filter_by_date(df, start, end)
        if df.empty:
            continue

        try:
            snap = compute_snapshot_df(df, sym)
            rows.append(snap)
        except Exception as e:
            # skip bad symbol silently or log if preferred
            continue

    if not rows:
        raise HTTPException(404, "No data for requested symbols/date range.")

    report = pd.DataFrame(rows)

    # Sort by daily movers when available
    if "Pct_1D" in report.columns and report["Pct_1D"].notna().any():
        report = report.sort_values(by="Pct_1D", ascending=False, na_position="last")
    else:
        report = report.sort_values(by="Symbol")

    if top and top > 0:
        report = report.head(top)

    return report.to_dict(orient="records")

@app.get("/signals/{symbol}")
def get_signals(symbol: str, start: Optional[str] = None, end: Optional[str] = None,
                rsi_low: float = 30, rsi_high: float = 70,
                move_1d_abs_gt: float = 3.0, gap_abs_gt: float = 2.0,
                near_52w_high_within: float = 1.0, near_52w_low_within: float = 1.0):
    """
    Detailed signals for one symbol (RSI, SMA cross, 52w breakout, 1D move, gap).
    Thresholds configurable via query params.
    """
    df = load_stock_data(symbol)
    if df is None or df.empty:
        fetched = fetch_and_save(symbol)
        df = fetched if fetched is not None else pd.DataFrame()
    if df.empty:
        raise HTTPException(404, "No data for symbol.")

    df = filter_by_date(df, start, end)
    if df.empty:
        raise HTTPException(404, "No data in selected date range.")

    sig = compute_signals_df(df)
    alerts = apply_alert_rules(sig, rsi_low, rsi_high, move_1d_abs_gt, gap_abs_gt,
                               near_52w_high_within, near_52w_low_within)
    return {
        "symbol": symbol.upper(),
        "date": pd.to_datetime(df["Date"]).max(),
        "signals": sig,
        "alerts": alerts
    }

@app.get("/alerts")
def get_alerts(symbols: str = Query(..., description="Comma-separated symbols"),
               start: Optional[str] = None, end: Optional[str] = None,
               rsi_low: float = 30, rsi_high: float = 70,
               move_1d_abs_gt: float = 3.0, gap_abs_gt: float = 2.0,
               near_52w_high_within: float = 1.0, near_52w_low_within: float = 1.0,
               top: int = 0):
    """
    Scan a watchlist and return only entries with at least one triggered alert.
    Optional 'top' will return the top N by |1D move|.
    """
    syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not syms:
        raise HTTPException(400, "No symbols provided.")

    results = []
    for sym in syms:
        df = load_stock_data(sym)
        if df is None or df.empty:
            fetched = fetch_and_save(sym)
            df = fetched if fetched is not None else pd.DataFrame()
        if df.empty:
            continue

        df = filter_by_date(df, start, end)
        if df.empty:
            continue

        sig = compute_signals_df(df)
        alerts = apply_alert_rules(sig, rsi_low, rsi_high, move_1d_abs_gt, gap_abs_gt,
                                   near_52w_high_within, near_52w_low_within)
        if alerts:
            results.append({
                "symbol": sym,
                "date": pd.to_datetime(df["Date"]).max(),
                "ret_1d_pct": sig.get("ret_1d_pct"),
                "alerts": alerts,
                "signals": sig
            })

    if not results:
        return []

    # If top requested, sort by absolute 1D move desc
    if top and top > 0:
        results = sorted(results, key=lambda x: (abs(x.get("ret_1d_pct") or 0)), reverse=True)[:top]

    return results

# ---- Rules: list all
@app.get("/rules", response_model=list[RuleOut])
def list_rules():
    q = text('SELECT * FROM alert_rules ORDER BY "Symbol"')
    df = pd.read_sql(q, engine)
    return df.to_dict(orient="records")

# ---- Rules: get one
@app.get("/rules/{symbol}", response_model=RuleOut)
def get_rule(symbol: str):
    q = text('SELECT * FROM alert_rules WHERE "Symbol" = :s')
    df = pd.read_sql(q, engine, params={"s": symbol.upper()})
    if df.empty:
        # Return defaults if none exist yet (optional behavior)
        defaults = {
            "Symbol": symbol.upper(),
            "rsi_low": 30.0, "rsi_high": 70.0,
            "move_1d_abs_gt": 3.0, "gap_abs_gt": 2.0,
            "near_52w_high_within": 1.0, "near_52w_low_within": 1.0,
            "updated_at": None
        }
        return defaults
    return df.iloc[0].to_dict()

# ---- Rules: upsert
@app.put("/rules/{symbol}", response_model=RuleOut)
def upsert_rule(symbol: str, rule: RuleIn):
    symu = symbol.upper()
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
                s=symu, rsi_low=rule.rsi_low, rsi_high=rule.rsi_high,
                m1d=rule.move_1d_abs_gt, gap=rule.gap_abs_gt,
                nh=rule.near_52w_high_within, nl=rule.near_52w_low_within
            ))
    else:
        with engine.begin() as conn:
            conn.execute(text('DELETE FROM alert_rules WHERE "Symbol" = :s'), {"s": symu})
            conn.execute(text("""
                INSERT INTO alert_rules ("Symbol","rsi_low","rsi_high","move_1d_abs_gt","gap_abs_gt","near_52w_high_within","near_52w_low_within","updated_at")
                VALUES (:s,:rsi_low,:rsi_high,:m1d,:gap,:nh,:nl, CURRENT_TIMESTAMP)
            """), dict(
                s=symu, rsi_low=rule.rsi_low, rsi_high=rule.rsi_high,
                m1d=rule.move_1d_abs_gt, gap=rule.gap_abs_gt,
                nh=rule.near_52w_high_within, nl=rule.near_52w_low_within
            ))

    return get_rule(symu)

# ---- Rules: delete
@app.delete("/rules/{symbol}")
def delete_rule(symbol: str):
    with engine.begin() as conn:
        conn.execute(text('DELETE FROM alert_rules WHERE "Symbol" = :s'), {"s": symbol.upper()})
    return {"status": "ok", "deleted": symbol.upper()}


@app.post("/notify/telegram")
def post_telegram_alerts(
    symbols: str = Query(..., description="Comma-separated symbols to scan"),
    start: Optional[str] = None,
    end: Optional[str] = None,
    rsi_low: float = 30, rsi_high: float = 70,
    move_1d_abs_gt: float = 3.0, gap_abs_gt: float = 2.0,
    near_52w_high_within: float = 1.0, near_52w_low_within: float = 1.0,
    top: int = 0,
    dry_run: bool = Body(False, description="If true, do not send to Telegram; just return the message")
):
    """
    Scans /alerts and pushes a formatted summary to Telegram.
    """
    # Reuse your /alerts logic directly:
    syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not syms:
        raise HTTPException(400, "No symbols provided.")

    results = []
    for sym in syms:
        df = load_stock_data(sym)
        if df is None or df.empty:
            fetched = fetch_and_save(sym)
            df = fetched if fetched is not None else pd.DataFrame()
        if df.empty:
            continue

        df = filter_by_date(df, start, end)
        if df.empty:
            continue

        sig = compute_signals_df(df)
        alerts = apply_alert_rules(sig, rsi_low, rsi_high, move_1d_abs_gt, gap_abs_gt,
                                   near_52w_high_within, near_52w_low_within)
        if alerts:
            results.append({
                "symbol": sym,
                "date": pd.to_datetime(df["Date"]).max(),
                "ret_1d_pct": sig.get("ret_1d_pct"),
                "alerts": alerts,
                "signals": sig
            })

    if not results:
        msg = "No alerts triggered."
        if dry_run:
            return {"sent": False, "message": msg, "alerts": []}
        send_telegram_message(msg)
        return {"sent": True, "message": msg, "alerts": []}

    if top and top > 0:
        results = sorted(results, key=lambda x: (abs(x.get("ret_1d_pct") or 0)), reverse=True)[:top]

    # Build a compact Telegram message
    lines = ["*QuantFlow Alerts* 🔔"]
    for r in results:
        s = r["symbol"]
        ret = r.get("ret_1d_pct")
        ret_str = f" ({ret:+.2f}% 1D)" if ret is not None else ""
        lines.append(f"*{s}*{ret_str}")
        for a in r["alerts"]:
            lines.append(f"  • {a}")
    text_msg = "\n".join(lines)

    if dry_run:
        return {"sent": False, "message": text_msg, "alerts": results}

    resp = send_telegram_message(text_msg)
    return {"sent": True, "telegram_response": resp, "alerts": results}
