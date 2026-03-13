"""
backtest.py — Real event-driven backtester using actual historical data.

What this does:
  1. Pulls 5 years of daily OHLCV via yfinance (free, real data)
  2. Pulls actual earnings dates + EPS estimates via yfinance
  3. For each earnings event: records price change at T+1d, T+3d, T+5d, T+10d
  4. Filters by volume spike (> 1.5x avg) to test if volume confirms the move
  5. Outputs a real statistical summary: avg return, win rate, max loss, Sharpe

Usage:
    python -m modules.backtest                        # run default 10 tickers
    python -m modules.backtest --tickers NVDA TSLA    # specific tickers
    python -m modules.backtest --tickers NVDA --years 3
    python -m modules.backtest --query               # interactive query mode

Output:
    data/backtest_results.json   — raw results
    reports/backtest_YYYY-MM-DD.html — visual report
"""

import json, sys, os
from datetime import date, datetime, timedelta
from collections import defaultdict

import yfinance as yf
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Default universe ───────────────────────────────────────────────────────
DEFAULT_TICKERS = [
    "NVDA", "TSLA", "AAPL", "META", "AMD",
    "MSFT", "AMZN", "GOOGL", "PLTR", "SOFI",
]

HOLD_PERIODS = [1, 3, 5, 10]   # days after event


# ══════════════════════════════════════════════════════════════════════════
# DATA LAYER
# ══════════════════════════════════════════════════════════════════════════

def fetch_price_history(ticker: str, years: int = 5) -> pd.DataFrame:
    """
    Pull daily OHLCV for `ticker` going back `years` years.
    Returns DataFrame indexed by date with columns:
      open, high, low, close, volume, avg_vol_20, vol_ratio
    """
    end   = datetime.today()
    start = end - timedelta(days=years * 365 + 30)  # extra buffer

    tk   = yf.Ticker(ticker)
    hist = tk.history(start=start, end=end, interval="1d", auto_adjust=True)

    if hist.empty or len(hist) < 50:
        return pd.DataFrame()

    hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
    hist.columns = [c.lower() for c in hist.columns]

    # 20-day average volume (excluding today)
    hist["avg_vol_20"] = hist["volume"].shift(1).rolling(20).mean()
    hist["vol_ratio"]  = hist["volume"] / hist["avg_vol_20"].replace(0, np.nan)

    # Running technicals
    hist["ma_20"] = hist["close"].rolling(20).mean()
    hist["ma_50"] = hist["close"].rolling(50).mean()
    hist["rsi"]   = _rsi(hist["close"], 14)

    return hist


def _rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def fetch_earnings_events(ticker: str, hist: pd.DataFrame) -> list[dict]:
    """
    Pull earnings dates from yfinance and match to price history.
    Returns list of event dicts with actual price moves.
    """
    tk = yf.Ticker(ticker)
    events = []

    try:
        # earnings_dates gives us date + EPS estimate + actual
        ed = tk.earnings_dates
        if ed is None or ed.empty:
            return []

        ed.index = pd.to_datetime(ed.index).tz_localize(None).normalize()
        ed = ed.sort_index()

    except Exception as e:
        print(f"  [{ticker}] earnings_dates error: {e}")
        return []

    dates = hist.index.tolist()

    for event_date, row in ed.iterrows():
        # Skip future dates
        if event_date > pd.Timestamp.today().normalize():
            continue

        # Find the trading day of or after the event
        # (earnings often announced after-hours → price reaction next day)
        trade_date = _next_trading_day(event_date, dates)
        if trade_date is None:
            continue

        try:
            # Price on event trading day
            t0_close = hist.loc[trade_date, "close"]
            t0_open  = hist.loc[trade_date, "open"]
            t0_vol   = hist.loc[trade_date, "volume"]
            t0_vr    = hist.loc[trade_date, "vol_ratio"]
            t0_ma20  = hist.loc[trade_date, "ma_20"]
            t0_ma50  = hist.loc[trade_date, "ma_50"]
            t0_rsi   = hist.loc[trade_date, "rsi"]

            # Previous close (day before event)
            prev_idx = dates.index(trade_date) - 1
            if prev_idx < 0:
                continue
            prev_close = hist.iloc[prev_idx]["close"]

            # EPS surprise
            eps_est    = row.get("EPS Estimate", np.nan)
            eps_actual = row.get("Reported EPS", np.nan)
            if pd.notna(eps_est) and pd.notna(eps_actual) and eps_est != 0:
                surprise_pct = (eps_actual - eps_est) / abs(eps_est) * 100
                beat_miss = "beat" if surprise_pct > 2 else "miss" if surprise_pct < -2 else "inline"
            else:
                surprise_pct = np.nan
                beat_miss = "unknown"

            # Forward returns at each hold period
            trade_idx = dates.index(trade_date)
            returns   = {}
            for days in HOLD_PERIODS:
                future_idx = trade_idx + days
                if future_idx < len(dates):
                    future_close  = hist.iloc[future_idx]["close"]
                    # Entry = next day open (realistic)
                    entry_idx = trade_idx + 1
                    if entry_idx < len(dates):
                        entry_price = hist.iloc[entry_idx]["open"]
                    else:
                        entry_price = t0_close
                    returns[f"t{days}d"] = round(
                        (future_close - entry_price) / entry_price * 100, 3
                    )
                else:
                    returns[f"t{days}d"] = np.nan

            # Gap at open on event day
            gap_pct = round((t0_open - prev_close) / prev_close * 100, 2)

            events.append({
                "ticker":       ticker,
                "event_date":   event_date.strftime("%Y-%m-%d"),
                "trade_date":   trade_date.strftime("%Y-%m-%d"),
                "eps_estimate": round(float(eps_est), 4) if pd.notna(eps_est) else None,
                "eps_actual":   round(float(eps_actual), 4) if pd.notna(eps_actual) else None,
                "surprise_pct": round(float(surprise_pct), 2) if pd.notna(surprise_pct) else None,
                "beat_miss":    beat_miss,
                "gap_pct":      gap_pct,
                "vol_ratio":    round(float(t0_vr), 2) if pd.notna(t0_vr) else None,
                "above_ma20":   bool(t0_close > t0_ma20) if pd.notna(t0_ma20) else None,
                "rsi_at_event": round(float(t0_rsi), 1) if pd.notna(t0_rsi) else None,
                "price_at_event": round(float(t0_close), 2),
                "above_ma50": bool(t0_close > t0_ma50) if pd.notna(t0_ma50) else None,
                **returns,
            })

        except Exception:
            continue

    return events


def _next_trading_day(event_dt: pd.Timestamp, trading_days: list) -> pd.Timestamp | None:
    """Find the same day or next available trading day."""
    for delta in range(4):
        candidate = event_dt + pd.Timedelta(days=delta)
        if candidate in trading_days:
            return candidate
    return None


# ══════════════════════════════════════════════════════════════════════════
# STATISTICS ENGINE
# ══════════════════════════════════════════════════════════════════════════

def compute_stats(events: list[dict], label: str = "all") -> dict:
    """
    Compute real statistics from a list of events.
    Returns a dict with avg, median, win_rate, max_loss, sharpe per hold period.
    """
    if not events:
        return {"n": 0, "label": label}

    stats = {"n": len(events), "label": label}

    for period in [f"t{d}d" for d in HOLD_PERIODS]:
        vals = [e[period] for e in events if e.get(period) is not None and not np.isnan(e[period])]
        if not vals:
            continue
        arr = np.array(vals)
        stats[period] = {
            "avg":      round(float(np.mean(arr)), 3),
            "median":   round(float(np.median(arr)), 3),
            "win_rate": round(float((arr > 0).mean() * 100), 1),
            "max_loss": round(float(arr.min()), 3),
            "max_gain": round(float(arr.max()), 3),
            "std":      round(float(arr.std()), 3),
            "sharpe":   round(float(np.mean(arr) / arr.std()), 3) if arr.std() > 0 else 0,
            "n":        len(vals),
        }

    return stats


def segment_events(events: list[dict]) -> dict:
    """
    Break events into meaningful sub-groups for comparison.
    Focuses on post-earnings signal families that are more useful than
    simple beat+volume alone.
    """
    segments = {}

    # ------------------------------------------------------------------
    # Baseline / core buckets
    # ------------------------------------------------------------------
    segments["all"] = compute_stats(events, label="All Earnings Events (baseline)")

    for bm in ["beat", "miss", "inline", "unknown"]:
        subset = [e for e in events if e.get("beat_miss") == bm]
        if subset:
            segments[f"earnings_{bm}"] = compute_stats(
                subset,
                label=f"Earnings {bm.title()}"
            )

    # ------------------------------------------------------------------
    # Original buckets (kept for continuity / comparison)
    # ------------------------------------------------------------------
    beat_vol = [
        e for e in events
        if e.get("beat_miss") == "beat"
        and e.get("vol_ratio") is not None
        and e["vol_ratio"] >= 1.5
    ]
    if beat_vol:
        segments["beat_vol_spike"] = compute_stats(
            beat_vol,
            label="Beat + Vol Spike ≥1.5x"
        )

    beat_vol_strong = [
        e for e in events
        if e.get("beat_miss") == "beat"
        and e.get("vol_ratio") is not None
        and e["vol_ratio"] >= 2.0
    ]
    if beat_vol_strong:
        segments["beat_vol_2x"] = compute_stats(
            beat_vol_strong,
            label="Beat + Vol Spike ≥2.0x"
        )

    beat_trend_ma20 = [
        e for e in events
        if e.get("beat_miss") == "beat"
        and e.get("above_ma20")
    ]
    if beat_trend_ma20:
        segments["beat_above_ma20"] = compute_stats(
            beat_trend_ma20,
            label="Beat + Above MA20"
        )

    strong_beat = [
        e for e in events
        if e.get("surprise_pct") is not None
        and e["surprise_pct"] > 10
    ]
    if strong_beat:
        segments["strong_beat_10pct"] = compute_stats(
            strong_beat,
            label="Strong Beat >10% Surprise"
        )

    miss_vol = [
        e for e in events
        if e.get("beat_miss") == "miss"
        and e.get("vol_ratio") is not None
        and e["vol_ratio"] >= 1.5
    ]
    if miss_vol:
        segments["miss_vol_spike"] = compute_stats(
            miss_vol,
            label="Miss + Vol Spike ≥1.5x"
        )

    gap_beat = [
        e for e in events
        if e.get("beat_miss") == "beat"
        and e.get("gap_pct") is not None
        and e["gap_pct"] > 2
    ]
    if gap_beat:
        segments["beat_gap_up"] = compute_stats(
            gap_beat,
            label="Beat + Gap Up >2%"
        )

    # ------------------------------------------------------------------
    # New buckets aligned with current research direction
    # ------------------------------------------------------------------

    # Strong beat, but not already chased too hard at the open
    strong_beat_no_gap = [
        e for e in events
        if e.get("surprise_pct") is not None
        and e["surprise_pct"] > 10
        and e.get("gap_pct") is not None
        and e["gap_pct"] <= 2
    ]
    if strong_beat_no_gap:
        segments["strong_beat_no_gap"] = compute_stats(
            strong_beat_no_gap,
            label="Strong Beat >10% + Gap ≤2%"
        )

    # Beat, but little/no opening gap (often more actionable than chasing)
    beat_gap_small = [
        e for e in events
        if e.get("beat_miss") == "beat"
        and e.get("gap_pct") is not None
        and e["gap_pct"] <= 1.0
    ]
    if beat_gap_small:
        segments["beat_gap_small"] = compute_stats(
            beat_gap_small,
            label="Beat + Gap ≤1%"
        )

    # Beat while not already overbought
    beat_rsi_lt_70 = [
        e for e in events
        if e.get("beat_miss") == "beat"
        and e.get("rsi_at_event") is not None
        and e["rsi_at_event"] < 70
    ]
    if beat_rsi_lt_70:
        segments["beat_rsi_lt_70"] = compute_stats(
            beat_rsi_lt_70,
            label="Beat + RSI <70"
        )

    # Miss with a negative opening gap: possible cleaner short-side behavior
    miss_gap_down = [
        e for e in events
        if e.get("beat_miss") == "miss"
        and e.get("gap_pct") is not None
        and e["gap_pct"] < 0
    ]
    if miss_gap_down:
        segments["miss_gap_down"] = compute_stats(
            miss_gap_down,
            label="Miss + Gap Down"
        )

    # Trend context using MA50, closer to a simple regime proxy
    beat_above_ma50 = [
        e for e in events
        if e.get("beat_miss") == "beat"
        and e.get("above_ma50")
    ]
    if beat_above_ma50:
        segments["beat_above_ma50"] = compute_stats(
            beat_above_ma50,
            label="Beat + Above MA50"
        )

    # Optional: strong beat + trend + no big gap
    strong_beat_trend_no_gap = [
        e for e in events
        if e.get("surprise_pct") is not None
        and e["surprise_pct"] > 10
        and e.get("gap_pct") is not None
        and e["gap_pct"] <= 2
        and e.get("above_ma50")
    ]
    if strong_beat_trend_no_gap:
        segments["strong_beat_trend_no_gap"] = compute_stats(
            strong_beat_trend_no_gap,
            label="Strong Beat >10% + Gap ≤2% + Above MA50"
        )

    return segments


# ══════════════════════════════════════════════════════════════════════════
# MAIN RUN
# ══════════════════════════════════════════════════════════════════════════

def run_backtest(tickers: list[str] = None, years: int = 5, verbose: bool = True) -> dict:
    """
    Full backtest pipeline. Returns results dict.
    """
    tickers = tickers or DEFAULT_TICKERS
    start_time = datetime.now()

    all_events  = []
    per_ticker  = {}
    failed      = []

    print(f"\n{'═'*55}")
    print(f"  BACKTEST ENGINE — {len(tickers)} tickers, {years} years")
    print(f"  Using real historical data via yfinance")
    print(f"{'═'*55}\n")

    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i:2d}/{len(tickers)}] {ticker:<6} ", end="", flush=True)

        try:
            hist = fetch_price_history(ticker, years=years)
            if hist.empty:
                print("⚠ no price data")
                failed.append(ticker)
                continue

            events = fetch_earnings_events(ticker, hist)
            if not events:
                print("⚠ no earnings events found")
                failed.append(ticker)
                continue

            # Per-ticker segments
            segs = segment_events(events)
            per_ticker[ticker] = {
                "events":   events,
                "segments": segs,
                "n_events": len(events),
            }
            all_events.extend(events)

            # Quick summary line
            beat_events = [e for e in events if e.get("beat_miss") == "beat"]
            beat_5d     = [e["t5d"] for e in beat_events if e.get("t5d") is not None]
            if beat_5d:
                avg5 = np.mean(beat_5d)
                wr5  = (np.array(beat_5d) > 0).mean() * 100
                print(f"✓ {len(events):2d} events | beat→T+5d avg {avg5:+.1f}%  WR {wr5:.0f}%")
            else:
                print(f"✓ {len(events):2d} events")

        except Exception as e:
            print(f"✗ error: {e}")
            failed.append(ticker)

    print(f"\n  Total events collected: {len(all_events)}")

    # Cross-ticker aggregated segments
    print("\n  Computing aggregate statistics...")
    aggregate = segment_events(all_events)

    results = {
        "meta": {
            "run_date":    date.today().isoformat(),
            "tickers":     tickers,
            "years":       years,
            "n_total":     len(all_events),
            "n_tickers":   len(per_ticker),
            "failed":      failed,
            "elapsed_sec": round((datetime.now() - start_time).total_seconds(), 1),
        },
        "aggregate":  aggregate,
        "per_ticker": {t: v["segments"] for t, v in per_ticker.items()},
        "raw_events": all_events,
    }

    # Save results
    out_dir  = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "backtest_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved → {out_path}")

    # Print summary
    _print_summary(aggregate, results["meta"])

    # Generate HTML report
    html_path = _generate_html_report(results)
    print(f"  HTML report → {html_path}")

    return results


# ══════════════════════════════════════════════════════════════════════════
# QUERY INTERFACE
# ══════════════════════════════════════════════════════════════════════════

def query(results: dict, ticker: str = None, segment: str = None) -> dict:
    """
    Query saved backtest results.

    Examples:
        query(results, ticker="NVDA", segment="beat_vol_spike")
        query(results, segment="beat_vol_spike")   # cross-ticker
    """
    if ticker:
        data = results["per_ticker"].get(ticker, {})
        return data.get(segment, data) if segment else data
    else:
        data = results["aggregate"]
        return data.get(segment, data) if segment else data


def load_results() -> dict | None:
    """Load previously saved backtest results."""
    path = os.path.join(os.path.dirname(__file__), "..", "data", "backtest_results.json")
    if not os.path.exists(path):
        print(f"No saved results at {path}. Run the backtest first.")
        return None
    with open(path) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════
# OUTPUT
# ══════════════════════════════════════════════════════════════════════════

def _print_summary(aggregate: dict, meta: dict):
    """Print clean terminal summary of aggregate results."""
    print(f"\n{'═'*55}")
    print(f"  AGGREGATE RESULTS — {meta['n_total']} earnings events")
    print(f"  Tickers: {', '.join(meta['tickers'])}")
    print(f"{'═'*55}")

    # Key segments to compare
    key_segs = [
    ("all", "Baseline (all earnings)"),
    ("earnings_beat", "Earnings beat"),
    ("beat_gap_small", "Beat + Gap ≤1%"),
    ("strong_beat_10pct", "Strong beat >10% surprise"),
    ("strong_beat_no_gap", "Strong beat >10% + Gap ≤2%"),
    ("beat_rsi_lt_70", "Beat + RSI <70"),
    ("beat_above_ma50", "Beat + Above MA50"),
    ("beat_vol_spike", "Beat + Vol ≥1.5x"),
    ("earnings_miss", "Earnings miss"),
    ("miss_gap_down", "Miss + Gap Down"),
    ("miss_vol_spike", "Miss + Vol ≥1.5x"),
]

    header = f"  {'Segment':<35} {'N':>4}  {'T+1d':>7}  {'T+3d':>7}  {'T+5d':>7}  {'WR5d':>6}"
    print(header)
    print(f"  {'─'*75}")

    for seg_key, label in key_segs:
        seg = aggregate.get(seg_key)
        if not seg or seg.get("n", 0) == 0:
            continue

        n    = seg["n"]
        t1   = seg.get("t1d", {}).get("avg",      "—")
        t3   = seg.get("t3d", {}).get("avg",      "—")
        t5   = seg.get("t5d", {}).get("avg",      "—")
        wr5  = seg.get("t5d", {}).get("win_rate", "—")

        def fmt(v):
            if v == "—": return f"{'—':>7}"
            return f"{v:>+7.2f}%"
        def fmtwr(v):
            if v == "—": return f"{'—':>6}"
            return f"{v:>5.1f}%"

        marker = " ◄" if seg_key == "beat_vol_spike" else ""
        print(f"  {label:<35} {n:>4}  {fmt(t1)}  {fmt(t3)}  {fmt(t5)}  {fmtwr(wr5)}{marker}")

    print(f"\n  * Entry assumed at T+1 open (realistic execution)")
    print(f"  * Win rate = % of trades with positive return at T+5d")
    print(f"  * Data: yfinance daily OHLCV, {meta['years']} years\n")


def _generate_html_report(results: dict) -> str:
    """Build an HTML report with tables and bar charts."""
    meta      = results["meta"]
    aggregate = results["aggregate"]
    today     = date.today().isoformat()

    seg_order = [
        ("all",               "Baseline — all earnings"),
        ("earnings_beat",     "Earnings beat"),
        ("beat_vol_spike",    "Beat + Vol ≥1.5x"),
        ("beat_vol_2x",       "Beat + Vol ≥2.0x"),
        ("beat_above_ma20",   "Beat + Above MA20"),
        ("strong_beat_10pct", "Strong beat >10% surprise"),
        ("beat_gap_up",       "Beat + Gap Up >2%"),
        ("earnings_miss",     "Earnings miss"),
        ("miss_vol_spike",    "Miss + Vol ≥1.5x"),
        ("earnings_inline",   "Inline (no surprise)"),
    ]

    # Main aggregate table rows
    table_rows = ""
    for seg_key, label in seg_order:
        seg = aggregate.get(seg_key)
        if not seg or seg.get("n", 0) == 0:
            continue
        n   = seg["n"]
        highlight = "background:#0f2a1a;border-left:2px solid #22c55e;" if seg_key == "beat_vol_spike" else ""

        def cell(period, field):
            v = seg.get(period, {}).get(field)
            if v is None: return "<td style='color:#334155'>—</td>"
            color = "#22c55e" if v > 0 else "#ef4444" if v < 0 else "#94a3b8"
            suffix = "%" if field in ("avg","median","max_loss","max_gain") else ("%"  if field == "win_rate" else "")
            sign   = "+" if v > 0 and field != "win_rate" else ""
            return f"<td style='color:{color};font-weight:600;font-family:monospace'>{sign}{v:.2f}{suffix}</td>"

        table_rows += f"""
<tr style="{highlight}">
  <td style="color:#e2e8f0;font-weight:{'700' if seg_key=='beat_vol_spike' else '400'}">{label}</td>
  <td style="color:#64748b;text-align:center">{n}</td>
  {cell("t1d","avg")}{cell("t3d","avg")}{cell("t5d","avg")}{cell("t10d","avg")}
  {cell("t5d","win_rate")}{cell("t5d","max_loss")}{cell("t5d","sharpe")}
</tr>"""

    # Per-ticker table
    ticker_rows = ""
    for ticker, segs in results["per_ticker"].items():
        beat = segs.get("earnings_beat", {})
        combo = segs.get("beat_vol_spike", {})
        n_beat  = beat.get("n", 0)
        n_combo = combo.get("n", 0)

        def tval(seg, period, field):
            v = seg.get(period, {}).get(field)
            if v is None: return "<td style='color:#334155'>—</td>"
            color = "#22c55e" if v > 0 else "#ef4444" if v < 0 else "#94a3b8"
            sign = "+" if v > 0 else ""
            return f"<td style='color:{color};font-family:monospace'>{sign}{v:.2f}%</td>"

        ticker_rows += f"""
<tr>
  <td style="color:#e2e8f0;font-weight:700">{ticker}</td>
  <td style="color:#64748b;text-align:center">{n_beat}</td>
  {tval(beat,"t1d","avg")}{tval(beat,"t5d","avg")}
  <td style="color:#64748b;text-align:center">{n_combo}</td>
  {tval(combo,"t1d","avg")}{tval(combo,"t5d","avg")}
  <td style="color:{'#22c55e' if combo.get('t5d',{}).get('win_rate',0)>=55 else '#ef4444'};font-family:monospace">
    {combo.get("t5d",{}).get("win_rate","—")}{'%' if combo.get("t5d",{}).get("win_rate") else ""}
  </td>
</tr>"""

    # Insight block
    combo_seg  = aggregate.get("beat_vol_spike", {})
    base_seg   = aggregate.get("all", {})
    combo_5d   = combo_seg.get("t5d", {}).get("avg", 0) or 0
    base_5d    = base_seg.get("t5d",  {}).get("avg", 0) or 0
    combo_wr   = combo_seg.get("t5d", {}).get("win_rate", 0) or 0
    combo_n    = combo_seg.get("n", 0)
    combo_loss = combo_seg.get("t5d", {}).get("max_loss", 0) or 0

    edge_color = "#22c55e" if combo_5d > base_5d + 0.5 else "#f59e0b"
    edge_label = "POSITIVE EDGE" if combo_5d > 1 and combo_wr > 55 else "WEAK / NO EDGE"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Backtest Report — {today}</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,sans-serif;background:#0b0f1a;color:#e2e8f0;
        padding:28px;max-width:1000px;margin:0 auto;line-height:1.5}}
  h1{{font-size:20px;font-weight:700;color:#f1f5f9;margin-bottom:4px}}
  h2{{font-size:12px;font-weight:600;color:#64748b;text-transform:uppercase;
      letter-spacing:.1em;margin:28px 0 12px}}
  .meta{{font-size:12px;color:#475569;margin-bottom:24px}}
  table{{width:100%;border-collapse:collapse;font-size:12px;margin-bottom:6px}}
  th{{font-size:9px;color:#475569;text-transform:uppercase;letter-spacing:.08em;
      padding:6px 10px;text-align:left;border-bottom:.5px solid #1e293b}}
  th.r{{text-align:right}}
  td{{padding:8px 10px;border-bottom:.5px solid #111827;vertical-align:middle}}
  td.r{{text-align:right}}
  .insight{{background:#111827;border:.5px solid #1e293b;border-radius:8px;padding:18px 22px;margin-bottom:24px}}
  .edge-label{{font-size:11px;font-weight:700;color:{edge_color};letter-spacing:.1em;margin-bottom:8px}}
  .edge-val{{font-size:28px;font-weight:700;color:{edge_color};font-family:monospace}}
  .note{{font-size:12px;color:#64748b;margin-top:8px;line-height:1.6}}
  .grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:16px 0}}
  .stat{{background:#0d1224;border:.5px solid #1e293b;border-radius:6px;padding:12px 14px}}
  .stat-label{{font-size:9px;color:#475569;letter-spacing:.08em;margin-bottom:5px}}
  .stat-val{{font-size:16px;font-weight:700;font-family:monospace}}
  .warn{{background:#1c1000;border-left:3px solid #f59e0b;border-radius:4px;
         padding:12px 16px;font-size:12px;color:#94a3b8;line-height:1.6;margin-top:16px}}
</style>
</head><body>

<h1>📊 Backtest Report — Real Historical Data</h1>
<div class="meta">
  Run: {today} &nbsp;·&nbsp;
  Tickers: {', '.join(meta['tickers'])} &nbsp;·&nbsp;
  Period: {meta['years']} years &nbsp;·&nbsp;
  Total events: {meta['n_total']} &nbsp;·&nbsp;
  Computed in: {meta['elapsed_sec']}s
</div>

<h2>Key Finding — Beat + Volume Spike Combo</h2>
<div class="insight">
  <div class="edge-label">{edge_label}</div>
  <div style="display:flex;align-items:baseline;gap:16px;margin-bottom:12px">
    <div class="edge-val">{combo_5d:+.2f}%</div>
    <div style="font-size:14px;color:#64748b">avg T+5d return &nbsp;(baseline: {base_5d:+.2f}%)</div>
  </div>
  <div class="grid">
    {''.join(f'''<div class="stat">
      <div class="stat-label">{lbl}</div>
      <div class="stat-val" style="color:{c}">{val}</div>
    </div>''' for lbl,val,c in [
      ("EVENTS (n)",       str(combo_n),                  "#94a3b8"),
      ("WIN RATE @ T+5d",  f"{combo_wr:.1f}%",            "#22c55e" if combo_wr>=55 else "#ef4444"),
      ("MAX LOSS @ T+5d",  f"{combo_loss:+.2f}%",         "#ef4444"),
      ("SHARPE (T+5d)",    str(combo_seg.get("t5d",{}).get("sharpe","—")), "#94a3b8"),
    ])}
  </div>
  <div class="note">
    <strong style="color:#e2e8f0">What this means:</strong>
    When a stock reports earnings beat <em>and</em> volume spikes ≥1.5× average,
    the average T+5d return across {combo_n} historical instances is
    <strong style="color:{edge_color}">{combo_5d:+.2f}%</strong> with a
    <strong style="color:{'#22c55e' if combo_wr>=55 else '#ef4444'}">{combo_wr:.1f}%</strong> win rate.
    {'This is a statistically interesting edge worth monitoring.' if combo_wr >= 55 and combo_5d > 0.5
     else 'Win rate below 55% or return too small — this combo does not show reliable edge in this dataset.'}
  </div>
  <div class="warn">
    ⚠ <strong>Limitations:</strong> Past returns do not guarantee future results.
    This data uses daily closes and T+1 open as entry — actual execution will differ due to slippage,
    gap risk, and changing market regimes. n={combo_n} events is {'sufficient' if combo_n >= 30 else 'small — interpret with caution'}.
  </div>
</div>

<h2>All Segments — Aggregate ({meta['n_total']} events)</h2>
<table>
  <thead><tr>
    <th>Segment</th><th>N</th>
    <th class="r">T+1d avg</th><th class="r">T+3d avg</th>
    <th class="r">T+5d avg</th><th class="r">T+10d avg</th>
    <th class="r">WR@5d</th><th class="r">MaxLoss</th><th class="r">Sharpe</th>
  </tr></thead>
  <tbody>{table_rows}</tbody>
</table>

<h2>Per-Ticker Breakdown</h2>
<table>
  <thead><tr>
    <th>Ticker</th>
    <th>n(beat)</th><th class="r">beat T+1d</th><th class="r">beat T+5d</th>
    <th>n(+vol)</th><th class="r">+vol T+1d</th><th class="r">+vol T+5d</th>
    <th class="r">WR@5d</th>
  </tr></thead>
  <tbody>{ticker_rows}</tbody>
</table>

<div style="margin-top:24px;font-size:11px;color:#334155;border-top:.5px solid #1e293b;padding-top:14px">
  Data source: Yahoo Finance (yfinance) · Entry: T+1 open · All returns exclude dividends
  · This is a personal research tool, not financial advice.
</div>
</body></html>"""

    report_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
    os.makedirs(report_dir, exist_ok=True)
    path = os.path.join(report_dir, f"backtest_{today}.html")
    with open(path, "w") as f:
        f.write(html)
    return path


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Alpha Engine — Real Backtester")
    p.add_argument("--tickers", nargs="+", default=None,
                   help="Tickers to backtest (default: 10 high-vol stocks)")
    p.add_argument("--years", type=int, default=5,
                   help="Years of history (default: 5)")
    p.add_argument("--query", action="store_true",
                   help="Query saved results interactively")
    p.add_argument("--segment", default=None,
                   help="Segment to query: beat_vol_spike, earnings_beat, etc.")
    p.add_argument("--ticker", default=None,
                   help="Filter query to single ticker")
    p.add_argument("--no-open", action="store_true")
    args = p.parse_args()

    if args.query:
        results = load_results()
        if results:
            data = query(results, ticker=args.ticker, segment=args.segment)
            print(json.dumps(data, indent=2))
    else:
        results = run_backtest(
            tickers=args.tickers,
            years=args.years,
        )
        # Open HTML report
        if not args.no_open:
            import subprocess, platform
            html = os.path.join(
                os.path.dirname(__file__), "..",
                "reports", f"backtest_{date.today().isoformat()}.html"
            )
            if os.path.exists(html):
                opener = "open" if platform.system() == "Darwin" else "xdg-open"
                subprocess.Popen([opener, html])
