#!/usr/bin/env python3
"""
backtest_runner.py — Price-layer historical backtest for Alpha Engine

Downloads ~90 trading days of OHLCV for all watchlist symbols via yfinance,
reconstructs daily price rows, and scores them using the same layers as the
live system. EventEdge and Freshness use fixed proxies (historical news
unavailable without a paid API).

Checks directional accuracy at t+1, t+3, t+5.
Direction proxy: above MA20 → LONG, below MA20 → SHORT.

Usage:
    python3 backtest_runner.py              # print results
    python3 backtest_runner.py --csv        # also save data/backtest.csv
    python3 backtest_runner.py --days 60    # shorter lookback
    python3 backtest_runner.py --verbose    # show per-symbol errors
"""

import argparse
import sys
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

from modules.db import get_conn
from modules.analyzer import (
    score_market_confirmation,
    score_regime_fit,
    score_relative_opportunity,
    score_risk_penalty,
    compute_final_score,
    determine_action,
)

# ── Fixed proxies for layers that require historical news ─────────────────────
# EventEdge (0-25): 15.0 = observed average from live runs
# Freshness  (0-10):  6.0 = mid-range, assumes some news exists
FIXED_EVENT_EDGE = 15.0
FIXED_FRESHNESS  = 6.0
STUB_NEWS        = {"mixedness": 0, "sentiment": 0.0}

# Need extra calendar days to get enough trading days (weekends/holidays)
DOWNLOAD_BUFFER = 180   # calendar days downloaded
MIN_BARS        = 60    # bars needed before scoring starts (RSI/MA stability)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_symbols(conn) -> List[str]:
    rows = conn.execute(
        "SELECT symbol FROM watched_symbols WHERE enabled=1"
    ).fetchall()
    return [r["symbol"] for r in rows]


def get_market_caps(conn) -> Dict[str, float]:
    rows = conn.execute(
        "SELECT symbol, market_cap FROM price_snapshots "
        "WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM price_snapshots)"
    ).fetchall()
    return {r["symbol"]: float(r["market_cap"] or 0) for r in rows}


def determine_regime(spy_chg: float, spy_rsi: float, spy_vr: float) -> str:
    if spy_chg > 1.0 and spy_rsi < 75:      return "bull"
    if spy_chg < -1.0 and spy_rsi > 25:     return "bear"
    if abs(spy_chg) < 0.3 and spy_vr < 0.8: return "choppy"
    return "neutral"


def precompute_indicators(hist: pd.DataFrame, market_cap: float) -> pd.DataFrame:
    """Pre-compute all rolling indicators for a symbol in one pass (O(n))."""
    close  = hist["Close"].astype(float)
    high   = hist["High"].astype(float)
    low    = hist["Low"].astype(float)
    volume = hist["Volume"].astype(float)

    # RSI(14)
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, 1e-9)
    rsi   = 100 - (100 / (1 + rs))

    # ATR(14)
    prev_c = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_c).abs(),
        (low  - prev_c).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    # Moving averages
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()

    # Volume ratio (prior 20-day avg vs today)
    avg_vol   = volume.rolling(20).mean().shift(1)
    vol_ratio = volume / avg_vol.replace(0, float("nan"))

    # 52-week high/low (trailing 252 bars, min 50)
    high_52 = close.rolling(252, min_periods=50).max()
    low_52  = close.rolling(252, min_periods=50).min()

    return pd.DataFrame({
        "close_price":  close,
        "change_pct":   close.pct_change() * 100,
        "volume":       volume,
        "avg_volume":   avg_vol,
        "volume_ratio": vol_ratio,
        "rsi_14":       rsi,
        "ma_20":        ma20,
        "ma_50":        ma50,
        "above_ma20":   (close > ma20).astype(int),
        "atr_14":       atr,
        "week_high_52": high_52,
        "week_low_52":  low_52,
        "market_cap":   market_cap,
    })


def row_to_price_dict(row: pd.Series) -> dict:
    """Convert a DataFrame row to a dict matching price_snapshots column names."""
    return {k: (None if pd.isna(v) else v) for k, v in row.items()}


# ── Core backtest ─────────────────────────────────────────────────────────────

def run_backtest(conn, lookback_days: int = 90, verbose: bool = False) -> pd.DataFrame:
    symbols  = get_symbols(conn)
    mktcaps  = get_market_caps(conn)

    from datetime import date, timedelta
    end_dt   = date.today()
    start_dt = end_dt - timedelta(days=DOWNLOAD_BUFFER)

    print(f"[backtest] downloading {DOWNLOAD_BUFFER} calendar days for {len(symbols)} symbols...")

    raw = yf.download(
        symbols,
        start=start_dt.isoformat(),
        end=end_dt.isoformat(),
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if raw.empty:
        print("[backtest] ERROR: yfinance returned no data.")
        return pd.DataFrame()

    # Handle both single-symbol (Series) and multi-symbol (DataFrame) returns
    def get_field(field: str, sym: str) -> Optional[pd.Series]:
        try:
            s = raw[field][sym] if sym in raw[field].columns else None
            return s.dropna() if s is not None else None
        except Exception:
            return None

    # Pre-compute indicators for every symbol
    print("[backtest] computing indicators...")
    indicators: Dict[str, pd.DataFrame] = {}

    for sym in symbols:
        close = get_field("Close", sym)
        if close is None or len(close) < MIN_BARS:
            continue
        hist = pd.DataFrame({
            "Close":  raw["Close"][sym],
            "High":   raw["High"][sym],
            "Low":    raw["Low"][sym],
            "Volume": raw["Volume"][sym],
        }).dropna()
        if len(hist) < MIN_BARS:
            continue
        indicators[sym] = precompute_indicators(hist, mktcaps.get(sym, 0))

    if "SPY" not in indicators:
        print("[backtest] ERROR: SPY data missing — cannot compute regime.")
        return pd.DataFrame()

    spy_ind = indicators["SPY"]

    # Determine which dates to score (trading days within lookback window)
    cutoff = pd.Timestamp(end_dt - timedelta(days=lookback_days))
    scoring_dates = spy_ind.index[spy_ind.index >= cutoff]
    # Reserve last 5 bars for t+5 forward returns
    scoring_dates = scoring_dates[:-5]

    print(f"[backtest] scoring {len(scoring_dates)} days × {len(indicators)-1} symbols...")

    results = []

    for dt in scoring_dates:
        spy_row = spy_ind.loc[dt]
        spy_chg = float(spy_row["change_pct"] or 0)
        spy_rsi = float(spy_row["rsi_14"] or 50)
        spy_vr  = float(spy_row["volume_ratio"] or 1)
        regime  = {
            "regime":     determine_regime(spy_chg, spy_rsi, spy_vr),
            "spy_change": spy_chg,
        }

        for sym, ind_df in indicators.items():
            if sym == "SPY":
                continue
            if dt not in ind_df.index:
                continue

            row = ind_df.loc[dt]
            if pd.isna(row["rsi_14"]) or pd.isna(row["ma_20"]):
                continue

            price_row = row_to_price_dict(row)
            direction = "LONG" if price_row["above_ma20"] else "SHORT"

            try:
                market_conf  = score_market_confirmation(price_row, direction, STUB_NEWS)
                regime_fit   = score_regime_fit(direction, regime, "general")
                rel_opp      = score_relative_opportunity(price_row)
                risk_penalty = score_risk_penalty(price_row, direction, STUB_NEWS, regime)

                final_score = compute_final_score(
                    event_edge_score    = FIXED_EVENT_EDGE,
                    market_conf_score   = market_conf,
                    regime_fit_score    = regime_fit,
                    relative_opp_score  = rel_opp,
                    freshness_score     = FIXED_FRESHNESS,
                    risk_penalty_score  = risk_penalty,
                )
                action = determine_action(final_score, direction, regime)
            except Exception as e:
                if verbose:
                    print(f"[backtest] score error {sym} {dt.date()}: {e}")
                continue

            # Forward returns
            future = ind_df[ind_df.index > dt]["close_price"]

            def fwd(n: int) -> Optional[float]:
                if len(future) < n:
                    return None
                fut_px = float(future.iloc[n - 1])
                entry  = float(price_row["close_price"])
                raw    = (fut_px - entry) / entry * 100
                return raw if direction == "LONG" else -raw

            t1, t3, t5 = fwd(1), fwd(3), fwd(5)

            results.append({
                "date":          dt.strftime("%Y-%m-%d"),
                "symbol":        sym,
                "direction":     direction,
                "action":        action,
                "score":         round(final_score, 1),
                "regime":        regime["regime"],
                "market_conf":   round(market_conf, 1),
                "regime_fit":    round(regime_fit, 1),
                "rel_opp":       round(rel_opp, 1),
                "risk_penalty":  round(risk_penalty, 1),
                "rsi":           round(float(row["rsi_14"]), 1),
                "entry_price":   round(float(price_row["close_price"]), 2),
                "t1_pnl":        round(t1, 3) if t1 is not None else None,
                "t3_pnl":        round(t3, 3) if t3 is not None else None,
                "t5_pnl":        round(t5, 3) if t5 is not None else None,
                "t1_win":        (1 if t1 > 0 else 0) if t1 is not None else None,
                "t3_win":        (1 if t3 > 0 else 0) if t3 is not None else None,
                "t5_win":        (1 if t5 > 0 else 0) if t5 is not None else None,
            })

    return pd.DataFrame(results)


# ── Output ────────────────────────────────────────────────────────────────────

def print_results(df: pd.DataFrame):
    resolved = df[df["t5_pnl"].notna()].copy()
    n = len(resolved)

    W = 65
    print(f"\n{'═' * W}")
    print(f"  Alpha Engine — Price Layer Backtest")
    print(f"  {df['date'].min()} → {df['date'].max()}")
    print(f"  {n:,} resolved signals · {df['symbol'].nunique()} symbols")
    print(f"{'═' * W}")

    # Overall win rates
    print(f"\n  Overall directional accuracy:")
    for label, col, pnl_col in [
        ("t+1", "t1_win", "t1_pnl"),
        ("t+3", "t3_win", "t3_pnl"),
        ("t+5", "t5_win", "t5_pnl"),
    ]:
        sub = df[df[col].notna()]
        wr  = sub[col].mean() * 100
        avg = sub[pnl_col].mean()
        print(f"    {label}: {wr:.1f}% win rate  avg {avg:+.2f}%  ({len(sub):,} signals)")

    # By action label
    print(f"\n{'─' * W}")
    print(f"  By Action Label  (t+5 directional)")
    print(f"  {'Label':<12} {'N':>6}  {'Win%':>6}  {'Avg P&L':>8}  {'Avg Score':>10}")
    print(f"  {'─' * 50}")
    for label in ["ACTIONABLE", "WATCHLIST", "MONITOR", "IGNORE"]:
        sub = resolved[resolved["action"] == label]
        if len(sub) == 0:
            continue
        wr  = sub["t5_win"].mean() * 100
        avg = sub["t5_pnl"].mean()
        sc  = sub["score"].mean()
        print(f"  {label:<12} {len(sub):>6,}  {wr:>5.1f}%  {avg:>+7.2f}%  {sc:>10.1f}")

    # By score bucket
    print(f"\n{'─' * W}")
    print(f"  By Score Bucket  (t+5 directional)")
    print(f"  {'Range':<12} {'N':>6}  {'Win%':>6}  {'Avg P&L':>8}")
    print(f"  {'─' * 40}")
    buckets = [(75, 101, "75+"), (65, 75, "65–74"), (55, 65, "55–64"),
               (45, 55, "45–54"), (0,  45, "< 45")]
    for lo, hi, lbl in buckets:
        sub = resolved[(resolved["score"] >= lo) & (resolved["score"] < hi)]
        if len(sub) == 0:
            continue
        wr  = sub["t5_win"].mean() * 100
        avg = sub["t5_pnl"].mean()
        print(f"  {lbl:<12} {len(sub):>6,}  {wr:>5.1f}%  {avg:>+7.2f}%")

    # By regime
    print(f"\n{'─' * W}")
    print(f"  By Regime  (t+5 directional)")
    print(f"  {'Regime':<10} {'N':>6}  {'Win%':>6}  {'Avg P&L':>8}")
    print(f"  {'─' * 38}")
    for reg in ["bull", "neutral", "bear", "choppy"]:
        sub = resolved[resolved["regime"] == reg]
        if len(sub) == 0:
            continue
        wr  = sub["t5_win"].mean() * 100
        avg = sub["t5_pnl"].mean()
        print(f"  {reg:<10} {len(sub):>6,}  {wr:>5.1f}%  {avg:>+7.2f}%")

    # By direction
    print(f"\n{'─' * W}")
    print(f"  By Direction  (t+5 directional)")
    print(f"  {'Direction':<10} {'N':>6}  {'Win%':>6}  {'Avg P&L':>8}")
    print(f"  {'─' * 38}")
    for d in ["LONG", "SHORT"]:
        sub = resolved[resolved["direction"] == d]
        if len(sub) == 0:
            continue
        wr  = sub["t5_win"].mean() * 100
        avg = sub["t5_pnl"].mean()
        print(f"  {d:<10} {len(sub):>6,}  {wr:>5.1f}%  {avg:>+7.2f}%")

    # Component correlation with t+5 P&L
    print(f"\n{'─' * W}")
    print(f"  Layer correlation with t+5 P&L")
    print(f"  {'Layer':<16} {'Pearson r':>10}  {'Interpretation'}")
    print(f"  {'─' * 52}")
    for col, name in [
        ("market_conf",  "MarketConf"),
        ("regime_fit",   "RegimeFit"),
        ("rel_opp",      "RelOpp"),
        ("risk_penalty", "RiskPenalty"),
        ("score",        "FinalScore"),
    ]:
        r = resolved[col].corr(resolved["t5_pnl"])
        interp = "strong" if abs(r) > 0.15 else "moderate" if abs(r) > 0.07 else "weak"
        sign   = "↑ helps" if r > 0 else "↓ hurts"
        print(f"  {name:<16} {r:>+9.3f}   {interp} {sign}")

    print(f"\n{'═' * W}")
    print(f"  Caveats:")
    print(f"  · EventEdge fixed at {FIXED_EVENT_EDGE}/25 (no historical news)")
    print(f"  · Direction from MA20 trend, not news sentiment")
    print(f"  · Validates price layers only — not full system")
    print(f"{'═' * W}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Alpha Engine price-layer backtest")
    parser.add_argument("--csv",     action="store_true", help="Save to data/backtest.csv")
    parser.add_argument("--days",    type=int, default=90, help="Lookback trading days (default 90)")
    parser.add_argument("--verbose", action="store_true", help="Show per-symbol errors")
    args = parser.parse_args()

    conn = get_conn()
    df   = run_backtest(conn, lookback_days=args.days, verbose=args.verbose)
    conn.close()

    if df.empty:
        print("No results. Check that watched_symbols has enabled symbols.")
        sys.exit(1)

    print_results(df)

    if args.csv:
        path = "data/backtest.csv"
        df.to_csv(path, index=False)
        print(f"[backtest] saved → {path}")


if __name__ == "__main__":
    main()
