"""
news_event_backtest.py — News-driven event study backtester

What this does:
  1. Loads historical news events from local SQLite (news_articles)
  2. Pulls historical daily OHLCV via yfinance
  3. Aligns each news event to a realistic entry date (default: next open)
  4. Computes forward returns at T+1d / T+3d / T+5d / T+10d
  5. Segments events by source quality, event type, sentiment, novelty, gap, and regime proxy
  6. Saves JSON + HTML report

Usage:
    python -m modules.news_event_backtest
    python -m modules.news_event_backtest --tickers NVDA AMD AAPL
    python -m modules.news_event_backtest --years 3 --no-open
    python -m modules.news_event_backtest --query --segment positive_small_gap
"""

import json
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "alpha.db")
DEFAULT_TICKERS = [
    "NVDA", "TSLA", "AAPL", "META", "AMD",
    "MSFT", "AMZN", "GOOGL", "PLTR", "SOFI",
]
HOLD_PERIODS = [1, 3, 5, 10]


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _to_row_dicts(rows) -> List[Dict[str, Any]]:
    return [dict(r) for r in rows]


def get_conn(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def is_low_value_title(title: str) -> bool:
    t = (title or "").lower()
    bad_patterns = [
        "here’s what",
        "here's what",
        "has to say about",
        "think about",
        "thinks about",
        "maintains a hold rating",
        "maintains a buy rating",
        "maintains a sell rating",
        "price prediction",
        "stock prediction",
        "is it a buy",
        "should you buy",
        "top stock to buy",
        "why this stock",
        "outlier as",
    ]
    return any(p in t for p in bad_patterns)


def source_tier(source: str) -> str:
    s = (source or "").strip().lower()
    if "reuters" in s or "sec" in s or "edgar" in s or "company ir" in s:
        return "high"
    if "associated press" in s or s == "ap" or "marketwatch" in s or "cnbc" in s or "yahoo finance" in s:
        return "medium"
    if "benzinga" in s or "seeking alpha" in s:
        return "low"
    return "unknown"


def source_weight(source: str) -> float:
    tier = source_tier(source)
    return {
        "high": 1.0,
        "medium": 0.7,
        "low": 0.35,
        "unknown": 0.4,
    }.get(tier, 0.4)


def normalize_event_type(article: Dict[str, Any]) -> str:
    evt = (article.get("event_type") or "general").lower()
    title = (article.get("title") or "").lower()
    content = (article.get("content") or "").lower()
    source = (article.get("source") or "").lower()
    text = f"{title} {content}"

    if any(x in text for x in ["gdp", "inflation", "cpi", "ppi", "fed", "powell", "rates", "rate cut", "rate hike"]):
        return "macro"

    if any(x in text for x in ["acquire", "acquisition", "merger", "buyout", "take private", "deal talks"]):
        return "ma"

    if any(x in text for x in ["sec", "investigation", "probe", "lawsuit", "settlement", "antitrust", "regulator", "doj", "ftc"]):
        return "regulation"

    if any(x in text for x in ["earnings", "eps", "revenue", "guidance", "quarterly results", "profit outlook"]):
        return "earnings"

    if any(x in text for x in ["partnership", "partners with", "multi-year partnership", "collaboration"]):
        return "partnership"

    if any(x in text for x in ["ai", "artificial intelligence", "model release", "chip launch"]):
        return "ai"

    if any(x in text for x in ["launch", "new product", "unveil", "released", "rollout"]):
        return "product"

    if any(x in text for x in ["layoff", "job cuts", "cuts workforce", "reduces headcount"]):
        return "layoffs"

    if any(x in text for x in ["price prediction", "stock prediction", "is it a buy", "should you buy", "opinion", "editorial"]):
        return "general"

    if "benzinga" in source and evt == "earnings":
        return "general"

    return evt


def sentiment_bucket(sent: float) -> str:
    if sent > 0.2:
        return "positive"
    if sent < -0.2:
        return "negative"
    return "neutral"


# ══════════════════════════════════════════════════════════════════════════
# News loading
# ══════════════════════════════════════════════════════════════════════════

def load_news_events(
    conn: sqlite3.Connection,
    tickers: Optional[List[str]] = None,
    years: int = 3,
) -> List[Dict[str, Any]]:
    """
    Load historical symbol-specific news from local DB.
    One news row may map to multiple symbols; we expand to one event per symbol.
    """
    start_date = (datetime.today() - timedelta(days=years * 365 + 30)).strftime("%Y-%m-%d")

    rows = conn.execute(
        """
        SELECT *
        FROM news_articles
        WHERE published_at >= ?
        ORDER BY published_at ASC
        """,
        (start_date,),
    ).fetchall()

    events: List[Dict[str, Any]] = []
    ticker_set = set(tickers or [])

    for row in _to_row_dicts(rows):
        title = row.get("title") or ""
        source = row.get("source") or ""

        # hard filters
        if is_low_value_title(title):
            continue
        if source_tier(source) == "low":
            continue

        try:
            syms = json.loads(row.get("symbols") or "[]")
        except Exception:
            syms = []

        # only keep clean ticker rows
        syms = [s.strip().upper() for s in syms if s and isinstance(s, str)]
        if tickers:
            syms = [s for s in syms if s in ticker_set]

        if not syms:
            continue

        norm_evt = normalize_event_type(row)

        # skip general low-information items for now
        if norm_evt == "general":
            continue

        for sym in syms[:3]:
            events.append({
                "news_id": row.get("id"),
                "symbol": sym,
                "published_at": row.get("published_at"),
                "source": source,
                "source_tier": source_tier(source),
                "event_type": norm_evt,
                "sentiment_score": _safe_float(row.get("sentiment_score"), 0.0),
                "importance_score": _safe_float(row.get("importance_score"), 0.3),
                "novelty_score": _safe_float(row.get("novelty_score"), 0.5),
                "title": title,
                "content": row.get("content") or "",
            })

    return events


# ══════════════════════════════════════════════════════════════════════════
# Price history
# ══════════════════════════════════════════════════════════════════════════

def fetch_price_history(ticker: str, years: int = 3) -> pd.DataFrame:
    end = datetime.today()
    start = end - timedelta(days=years * 365 + 40)

    tk = yf.Ticker(ticker)
    hist = tk.history(start=start, end=end, interval="1d", auto_adjust=True)

    if hist.empty or len(hist) < 60:
        return pd.DataFrame()

    hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
    hist.columns = [c.lower() for c in hist.columns]

    hist["avg_vol_20"] = hist["volume"].shift(1).rolling(20).mean()
    hist["vol_ratio"] = hist["volume"] / hist["avg_vol_20"].replace(0, np.nan)
    hist["ma_20"] = hist["close"].rolling(20).mean()
    hist["ma_50"] = hist["close"].rolling(50).mean()
    hist["rsi"] = _rsi(hist["close"], 14)

    tr1 = hist["high"] - hist["low"]
    tr2 = (hist["high"] - hist["close"].shift(1)).abs()
    tr3 = (hist["low"] - hist["close"].shift(1)).abs()
    hist["atr_14"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()

    return hist


# ══════════════════════════════════════════════════════════════════════════
# Event alignment and sample building
# ══════════════════════════════════════════════════════════════════════════

def align_event_to_trade_date(published_at: str, trading_days: List[pd.Timestamp]) -> Optional[Dict[str, Any]]:
    """
    Simplified alignment using local calendar time.
    Assumptions:
      - premarket: entry same-day open
      - intraday: entry next-day open
      - afterhours: entry next-day open
    """
    if not published_at:
        return None

    try:
        dt = pd.to_datetime(published_at).tz_localize(None)
    except Exception:
        try:
            dt = pd.to_datetime(str(published_at).replace("Z", "")).tz_localize(None)
        except Exception:
            return None

    event_day = dt.normalize()
    hhmm = dt.hour * 100 + dt.minute

    if hhmm < 930:
        timing_bucket = "premarket"
        candidate_entry_day = event_day
    elif hhmm <= 1600:
        timing_bucket = "intraday"
        candidate_entry_day = event_day + pd.Timedelta(days=1)
    else:
        timing_bucket = "afterhours"
        candidate_entry_day = event_day + pd.Timedelta(days=1)

    entry_date = _next_trading_day(candidate_entry_day, trading_days)
    if entry_date is None:
        return None

    trade_date = entry_date
    return {
        "event_day": event_day,
        "entry_date": entry_date,
        "trade_date": trade_date,
        "timing_bucket": timing_bucket,
    }


def _next_trading_day(start_dt: pd.Timestamp, trading_days: List[pd.Timestamp]) -> Optional[pd.Timestamp]:
    for delta in range(0, 5):
        candidate = start_dt + pd.Timedelta(days=delta)
        if candidate in trading_days:
            return candidate
    return None


def build_event_samples(news_events: List[Dict[str, Any]], years: int = 3, verbose: bool = True) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in news_events:
        grouped[e["symbol"]].append(e)

    all_samples: List[Dict[str, Any]] = []

    for i, (ticker, events) in enumerate(grouped.items(), 1):
        if verbose:
            print(f"  [{i:2d}/{len(grouped)}] {ticker:<6} loading prices...", end="", flush=True)

        hist = fetch_price_history(ticker, years=years)
        if hist.empty:
            if verbose:
                print(" no price data")
            continue

        dates = hist.index.tolist()

        samples_for_ticker = 0
        for e in events:
            aligned = align_event_to_trade_date(e["published_at"], dates)
            if aligned is None:
                continue

            entry_date = aligned["entry_date"]
            if entry_date not in hist.index:
                continue

            entry_idx = dates.index(entry_date)
            if entry_idx >= len(dates):
                continue

            try:
                entry_open = float(hist.loc[entry_date, "open"])
                event_close = float(hist.loc[entry_date, "close"])
                event_vol_ratio = _safe_float(hist.loc[entry_date, "vol_ratio"], np.nan)
                event_rsi = _safe_float(hist.loc[entry_date, "rsi"], np.nan)
                ma20 = _safe_float(hist.loc[entry_date, "ma_20"], np.nan)
                ma50 = _safe_float(hist.loc[entry_date, "ma_50"], np.nan)
                prev_idx = entry_idx - 1
                prev_close = float(hist.iloc[prev_idx]["close"]) if prev_idx >= 0 else event_close
                gap_pct = round((entry_open - prev_close) / prev_close * 100, 2) if prev_close else None

                # forward returns
                returns = {}
                for d in HOLD_PERIODS:
                    future_idx = entry_idx + d
                    if future_idx < len(dates):
                        future_close = float(hist.iloc[future_idx]["close"])
                        returns[f"t{d}d"] = round((future_close - entry_open) / entry_open * 100, 3)
                    else:
                        returns[f"t{d}d"] = np.nan

                sample = {
                    "symbol": ticker,
                    "news_id": e["news_id"],
                    "title": e["title"],
                    "source": e["source"],
                    "source_tier": e["source_tier"],
                    "event_type": e["event_type"],
                    "published_at": e["published_at"],
                    "trade_date": entry_date.strftime("%Y-%m-%d"),
                    "timing_bucket": aligned["timing_bucket"],
                    "sentiment_score": e["sentiment_score"],
                    "sentiment_bucket": sentiment_bucket(e["sentiment_score"]),
                    "importance_score": e["importance_score"],
                    "novelty_score": e["novelty_score"],
                    "entry_price": round(entry_open, 3),
                    "price_at_event_close": round(event_close, 3),
                    "gap_pct": gap_pct,
                    "vol_ratio": round(float(event_vol_ratio), 2) if pd.notna(event_vol_ratio) else None,
                    "rsi_at_event": round(float(event_rsi), 1) if pd.notna(event_rsi) else None,
                    "above_ma20": bool(event_close > ma20) if pd.notna(ma20) else None,
                    "above_ma50": bool(event_close > ma50) if pd.notna(ma50) else None,
                    "regime_proxy": "bull" if pd.notna(ma50) and event_close > ma50 else "bear",
                    **returns,
                }
                all_samples.append(sample)
                samples_for_ticker += 1
            except Exception:
                continue

        if verbose:
            print(f" {samples_for_ticker} events")

    return all_samples


# ══════════════════════════════════════════════════════════════════════════
# Statistics
# ══════════════════════════════════════════════════════════════════════════

def compute_stats(events: List[Dict[str, Any]], label: str = "all") -> Dict[str, Any]:
    if not events:
        return {"n": 0, "label": label}

    stats: Dict[str, Any] = {"n": len(events), "label": label}

    for period in [f"t{d}d" for d in HOLD_PERIODS]:
        vals = [e[period] for e in events if e.get(period) is not None and not np.isnan(e[period])]
        if not vals:
            continue

        arr = np.array(vals)
        stats[period] = {
            "avg": round(float(np.mean(arr)), 3),
            "median": round(float(np.median(arr)), 3),
            "win_rate": round(float((arr > 0).mean() * 100), 1),
            "loss_rate": round(float((arr < 0).mean() * 100), 1),
            "gt_3pct_rate": round(float((arr > 3).mean() * 100), 1),
            "max_loss": round(float(arr.min()), 3),
            "max_gain": round(float(arr.max()), 3),
            "std": round(float(arr.std()), 3),
            "sharpe": round(float(np.mean(arr) / arr.std()), 3) if arr.std() > 0 else 0.0,
            "n": len(vals),
        }

    return stats


def segment_events(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    segments: Dict[str, Any] = {}

    segments["all"] = compute_stats(events, label="All news events")

    # sentiment
    pos = [e for e in events if e["sentiment_score"] > 0.2]
    neg = [e for e in events if e["sentiment_score"] < -0.2]
    neu = [e for e in events if -0.2 <= e["sentiment_score"] <= 0.2]

    if pos:
        segments["positive_news"] = compute_stats(pos, label="Positive news")
    if neg:
        segments["negative_news"] = compute_stats(neg, label="Negative news")
    if neu:
        segments["neutral_news"] = compute_stats(neu, label="Neutral news")

    # event type
    for evt in ["product", "ai", "regulation", "partnership", "ma", "earnings", "layoffs", "macro"]:
        subset = [e for e in events if e["event_type"] == evt]
        if subset:
            segments[f"event_{evt}"] = compute_stats(subset, label=f"Event: {evt}")

    # source quality
    for tier in ["high", "medium", "low", "unknown"]:
        subset = [e for e in events if e["source_tier"] == tier]
        if subset:
            segments[f"source_{tier}"] = compute_stats(subset, label=f"Source: {tier}")

    # novelty / gap / RSI / regime
    pos_high_nov = [e for e in events if e["sentiment_score"] > 0.2 and e["novelty_score"] >= 0.8]
    if pos_high_nov:
        segments["positive_high_novelty"] = compute_stats(pos_high_nov, label="Positive + High Novelty")

    pos_small_gap = [
        e for e in events
        if e["sentiment_score"] > 0.2 and e.get("gap_pct") is not None and e["gap_pct"] <= 1.0
    ]
    if pos_small_gap:
        segments["positive_small_gap"] = compute_stats(pos_small_gap, label="Positive + Gap ≤1%")

    pos_rsi_ok = [
        e for e in events
        if e["sentiment_score"] > 0.2 and e.get("rsi_at_event") is not None and e["rsi_at_event"] < 70
    ]
    if pos_rsi_ok:
        segments["positive_rsi_lt_70"] = compute_stats(pos_rsi_ok, label="Positive + RSI <70")

    neg_gap_down = [
        e for e in events
        if e["sentiment_score"] < -0.2 and e.get("gap_pct") is not None and e["gap_pct"] < 0
    ]
    if neg_gap_down:
        segments["negative_gap_down"] = compute_stats(neg_gap_down, label="Negative + Gap Down")

    pos_bull = [e for e in events if e["sentiment_score"] > 0.2 and e.get("regime_proxy") == "bull"]
    pos_bear = [e for e in events if e["sentiment_score"] > 0.2 and e.get("regime_proxy") == "bear"]
    if pos_bull:
        segments["positive_bull_regime"] = compute_stats(pos_bull, label="Positive news in bull regime")
    if pos_bear:
        segments["positive_bear_regime"] = compute_stats(pos_bear, label="Positive news in bear regime")

    neg_bear = [e for e in events if e["sentiment_score"] < -0.2 and e.get("regime_proxy") == "bear"]
    if neg_bear:
        segments["negative_bear_regime"] = compute_stats(neg_bear, label="Negative news in bear regime")

    return segments


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def run_backtest(
    tickers: Optional[List[str]] = None,
    years: int = 3,
    db_path: str = DEFAULT_DB_PATH,
    verbose: bool = True,
) -> Dict[str, Any]:
    tickers = tickers or DEFAULT_TICKERS
    start_time = datetime.now()

    print(f"\n{'═'*60}")
    print(f"  NEWS EVENT BACKTEST — {len(tickers)} tickers, {years} years")
    print(f"  Using local DB news + yfinance daily OHLCV")
    print(f"{'═'*60}\n")

    conn = get_conn(db_path)
    news_events = load_news_events(conn, tickers=tickers, years=years)
    conn.close()

    if verbose:
        print(f"Loaded {len(news_events)} usable news events from DB\n")

    samples = build_event_samples(news_events, years=years, verbose=verbose)
    aggregate = segment_events(samples)

    per_ticker: Dict[str, Any] = {}
    for t in tickers:
        ts = [e for e in samples if e["symbol"] == t]
        if ts:
            per_ticker[t] = segment_events(ts)

    results = {
        "meta": {
            "run_date": date.today().isoformat(),
            "tickers": tickers,
            "years": years,
            "n_news_events": len(news_events),
            "n_samples": len(samples),
            "elapsed_sec": round((datetime.now() - start_time).total_seconds(), 1),
        },
        "aggregate": aggregate,
        "per_ticker": per_ticker,
        "raw_events": samples,
    }

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "news_event_backtest_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved → {out_path}")
    _print_summary(results["aggregate"], results["meta"])

    html_path = _generate_html_report(results)
    print(f"HTML report → {html_path}")

    return results


# ══════════════════════════════════════════════════════════════════════════
# Query
# ══════════════════════════════════════════════════════════════════════════

def query(results: Dict[str, Any], ticker: Optional[str] = None, segment: Optional[str] = None) -> Dict[str, Any]:
    if ticker:
        data = results["per_ticker"].get(ticker, {})
        return data.get(segment, data) if segment else data
    data = results["aggregate"]
    return data.get(segment, data) if segment else data


def load_results() -> Optional[Dict[str, Any]]:
    path = os.path.join(os.path.dirname(__file__), "..", "data", "news_event_backtest_results.json")
    if not os.path.exists(path):
        print(f"No saved results at {path}. Run the backtest first.")
        return None
    with open(path) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════
# Output
# ══════════════════════════════════════════════════════════════════════════

def _print_summary(aggregate: Dict[str, Any], meta: Dict[str, Any]) -> None:
    print(f"\n{'═'*60}")
    print(f"  AGGREGATE RESULTS — {meta['n_samples']} aligned news events")
    print(f"  Tickers: {', '.join(meta['tickers'])}")
    print(f"{'═'*60}")

    key_segs = [
        ("all", "Baseline (all news)"),
        ("positive_news", "Positive news"),
        ("negative_news", "Negative news"),
        ("positive_high_novelty", "Positive + High Novelty"),
        ("positive_small_gap", "Positive + Gap ≤1%"),
        ("positive_rsi_lt_70", "Positive + RSI <70"),
        ("positive_bull_regime", "Positive in bull regime"),
        ("positive_bear_regime", "Positive in bear regime"),
        ("negative_gap_down", "Negative + Gap Down"),
        ("source_high", "High-quality source"),
        ("source_low", "Low-quality source"),
    ]

    print(f"  {'Segment':<35} {'N':>4}  {'T+1d':>8}  {'T+3d':>8}  {'T+5d':>8}  {'WR5d':>6}")
    print(f"  {'─'*80}")

    for seg_key, label in key_segs:
        seg = aggregate.get(seg_key)
        if not seg or seg.get("n", 0) == 0:
            continue

        n = seg["n"]
        t1 = seg.get("t1d", {}).get("avg")
        t3 = seg.get("t3d", {}).get("avg")
        t5 = seg.get("t5d", {}).get("avg")
        wr5 = seg.get("t5d", {}).get("win_rate")

        def fmt(v):
            if v is None:
                return f"{'—':>8}"
            return f"{v:+8.2f}%"

        def fmtwr(v):
            if v is None:
                return f"{'—':>6}"
            return f"{v:>5.1f}%"

        print(f"  {label:<35} {n:>4}  {fmt(t1)}  {fmt(t3)}  {fmt(t5)}  {fmtwr(wr5)}")

    print("\n  * Entry assumed at next realistic open based on publish time bucket")
    print("  * Daily OHLCV only; not an intraday execution simulator\n")


def _generate_html_report(results: Dict[str, Any]) -> str:
    meta = results["meta"]
    aggregate = results["aggregate"]
    today = date.today().isoformat()

    seg_order = [
        ("all", "Baseline — all news"),
        ("positive_news", "Positive news"),
        ("negative_news", "Negative news"),
        ("positive_high_novelty", "Positive + High Novelty"),
        ("positive_small_gap", "Positive + Gap ≤1%"),
        ("positive_rsi_lt_70", "Positive + RSI <70"),
        ("positive_bull_regime", "Positive in bull regime"),
        ("positive_bear_regime", "Positive in bear regime"),
        ("negative_gap_down", "Negative + Gap Down"),
        ("source_high", "High-quality source"),
        ("source_medium", "Medium-quality source"),
        ("source_low", "Low-quality source"),
        ("event_product", "Event: product"),
        ("event_ai", "Event: ai"),
        ("event_partnership", "Event: partnership"),
        ("event_regulation", "Event: regulation"),
        ("event_macro", "Event: macro"),
    ]

    rows = ""
    for seg_key, label in seg_order:
        seg = aggregate.get(seg_key)
        if not seg or seg.get("n", 0) == 0:
            continue

        def cell(period, field):
            v = seg.get(period, {}).get(field)
            if v is None:
                return "<td style='color:#334155'>—</td>"
            color = "#22c55e" if v > 0 else "#ef4444" if v < 0 else "#94a3b8"
            suffix = "%" if field in ("avg", "median", "max_loss", "max_gain", "win_rate", "loss_rate", "gt_3pct_rate") else ""
            sign = "+" if v > 0 and field not in ("win_rate", "loss_rate", "gt_3pct_rate") else ""
            return f"<td style='color:{color};font-weight:600;font-family:monospace'>{sign}{v:.2f}{suffix}</td>"

        rows += f"""
<tr>
  <td style="color:#e2e8f0">{label}</td>
  <td style="color:#64748b;text-align:center">{seg["n"]}</td>
  {cell("t1d","avg")}
  {cell("t3d","avg")}
  {cell("t5d","avg")}
  {cell("t10d","avg")}
  {cell("t5d","win_rate")}
  {cell("t5d","max_loss")}
  {cell("t5d","sharpe")}
</tr>
"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>News Event Backtest — {today}</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,sans-serif;background:#0b0f1a;color:#e2e8f0;padding:28px;max-width:1000px;margin:0 auto;line-height:1.5}}
  h1{{font-size:20px;font-weight:700;color:#f1f5f9;margin-bottom:4px}}
  h2{{font-size:12px;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:.1em;margin:28px 0 12px}}
  .meta{{font-size:12px;color:#475569;margin-bottom:24px}}
  table{{width:100%;border-collapse:collapse;font-size:12px}}
  th{{font-size:9px;color:#475569;text-transform:uppercase;letter-spacing:.08em;padding:6px 10px;text-align:left;border-bottom:.5px solid #1e293b}}
  td{{padding:8px 10px;border-bottom:.5px solid #111827}}
  .note{{margin-top:24px;font-size:11px;color:#334155;border-top:.5px solid #1e293b;padding-top:14px}}
</style>
</head>
<body>
<h1>📰 News Event Backtest Report</h1>
<div class="meta">
  Run: {today} · Tickers: {", ".join(meta["tickers"])} · Period: {meta["years"]} years ·
  Raw news events: {meta["n_news_events"]} · Aligned samples: {meta["n_samples"]} ·
  Computed in: {meta["elapsed_sec"]}s
</div>

<h2>Aggregate Segments</h2>
<table>
  <thead>
    <tr>
      <th>Segment</th><th>N</th>
      <th>T+1d</th><th>T+3d</th><th>T+5d</th><th>T+10d</th>
      <th>WR@5d</th><th>MaxLoss</th><th>Sharpe</th>
    </tr>
  </thead>
  <tbody>
    {rows}
  </tbody>
</table>

<div class="note">
  Local DB news + Yahoo Finance daily OHLCV. Entry is aligned to a realistic next open based on publish time bucket.
  This is a personal research tool, not financial advice.
</div>
</body>
</html>
"""

    report_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
    os.makedirs(report_dir, exist_ok=True)
    path = os.path.join(report_dir, f"news_event_backtest_{today}.html")
    with open(path, "w") as f:
        f.write(html)
    return path


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import platform
    import subprocess

    p = argparse.ArgumentParser(description="Alpha Engine — News Event Backtester")
    p.add_argument("--tickers", nargs="+", default=None, help="Tickers to backtest")
    p.add_argument("--years", type=int, default=3, help="Years of history")
    p.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Path to SQLite DB")
    p.add_argument("--query", action="store_true", help="Query saved results")
    p.add_argument("--segment", default=None, help="Segment to query")
    p.add_argument("--ticker", default=None, help="Single ticker to query")
    p.add_argument("--no-open", action="store_true")
    args = p.parse_args()

    if args.query:
        results = load_results()
        if results:
            data = query(results, ticker=args.ticker, segment=args.segment)
            print(json.dumps(data, indent=2))
    else:
        run_backtest(
            tickers=args.tickers,
            years=args.years,
            db_path=args.db_path,
        )

        if not args.no_open:
            html = os.path.join(
                os.path.dirname(__file__), "..",
                "reports", f"news_event_backtest_{date.today().isoformat()}.html"
            )
            if os.path.exists(html):
                opener = "open" if platform.system() == "Darwin" else "xdg-open"
                subprocess.Popen([opener, html])