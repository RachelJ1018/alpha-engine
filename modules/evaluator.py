"""
evaluator.py — evaluation reports for Alpha Engine

 1. signal_stability_report(conn, days)    → signal volume, score distribution, event mix
 2. score_return_buckets(conn)             → does higher score → better return?
 3. paper_trade_summary(conn)              → exit distribution, win rate, P&L stats
 4. event_type_breakdown(conn)             → per-catalyst win rate and P&L
 5. weight_calibration_suggestions(conn)   → raise/lower event weights
 6. benchmark_adjusted_return(conn)        → dir-adj alpha vs SPY / sector ETF
 7. deduplicated_event_return(conn)        → collapse same ticker+event+week duplicates
 8. r_multiple_analysis(conn)              → P&L in units of ATR risk (R-multiples)
 9. win_rate_confidence_intervals(conn)    → Wilson 95% CI on win rate per event type
10. component_correlation_report(conn)     → Pearson r: each score component vs outcomes
11. rescore_comparison_report(conn)        → old vs new score quality after scoring rewrite
12. promoted_signal_quality_report(conn)   → quality of signals that gained score in rescore
13. false_upgrade_diagnosis_report(conn)   → Δscore≥3 but t5_pnl<0: find false-positive patterns
14. empirical_threshold_backtest(conn)     → win rate / alpha / R at each score threshold cut

All functions accept a sqlite3 connection (row_factory=sqlite3.Row) and return
plain dicts/lists safe to render directly in Streamlit.
"""

import math
import statistics
from datetime import date, timedelta


def _safe_avg(values):
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def _safe_median(values):
    vals = [v for v in values if v is not None]
    return round(statistics.median(vals), 2) if vals else None


# ─────────────────────────────────────────────────────────────────────────────
# 1. Signal Stability
# ─────────────────────────────────────────────────────────────────────────────

def signal_stability_report(conn, days=30) -> dict:
    """
    Answers: Is the system generating stable, consistent signals?

    Returns:
      days_covered, total_signals, avg_per_day
      action_counts        {ACTIONABLE, WATCHLIST, MONITOR, IGNORE}
      avg_score, median_score
      score_by_action      {action: {avg, min, max, count}}
      strategy_distribution {bucket: count}
      score_components_avg  {component: avg}
      daily_counts         [{date, total, ACTIONABLE, WATCHLIST, MONITOR, IGNORE}]
    """
    cutoff = (date.today() - timedelta(days=days)).isoformat()

    rows = conn.execute("""
        SELECT run_date, action, final_score, direction, strategy_bucket,
               event_edge_score, market_conf_score, regime_fit_score,
               relative_opp_score, freshness_score, risk_penalty_score
        FROM trade_candidates
        WHERE run_date >= ?
        ORDER BY run_date DESC
    """, (cutoff,)).fetchall()

    if not rows:
        return {"days_covered": 0, "total_signals": 0, "avg_per_day": 0}

    rows = [dict(r) for r in rows]

    # Exclude weekends (non-trading days)
    from datetime import date as _date
    rows = [r for r in rows if _date.fromisoformat(r["run_date"]).weekday() < 5]

    if not rows:
        return {"days_covered": 0, "total_signals": 0, "avg_per_day": 0}

    # Action counts
    action_counts = {"ACTIONABLE": 0, "WATCHLIST": 0, "MONITOR": 0, "IGNORE": 0}
    for r in rows:
        a = r.get("action") or "IGNORE"
        action_counts[a] = action_counts.get(a, 0) + 1

    # Score stats overall
    all_scores = [r["final_score"] for r in rows if r["final_score"] is not None]

    # Score per action level
    score_by_action = {}
    for action in ("ACTIONABLE", "WATCHLIST", "MONITOR", "IGNORE"):
        scores = [r["final_score"] for r in rows
                  if r.get("action") == action and r["final_score"] is not None]
        if scores:
            score_by_action[action] = {
                "avg":   _safe_avg(scores),
                "min":   round(min(scores), 1),
                "max":   round(max(scores), 1),
                "count": len(scores),
            }

    # Strategy bucket distribution
    strategy_dist = {}
    for r in rows:
        b = r.get("strategy_bucket") or "unknown"
        strategy_dist[b] = strategy_dist.get(b, 0) + 1

    # Daily counts
    daily_map = {}
    for r in rows:
        d = r["run_date"]
        if d not in daily_map:
            daily_map[d] = {"date": d, "total": 0,
                            "ACTIONABLE": 0, "WATCHLIST": 0,
                            "MONITOR": 0, "IGNORE": 0}
        daily_map[d]["total"] += 1
        a = r.get("action") or "IGNORE"
        daily_map[d][a] = daily_map[d].get(a, 0) + 1

    daily_counts = sorted(daily_map.values(), key=lambda x: x["date"])
    days_covered = len(daily_map)

    # Score component averages
    comp_fields = [
        "event_edge_score", "market_conf_score", "regime_fit_score",
        "relative_opp_score", "freshness_score", "risk_penalty_score",
    ]
    score_components_avg = {
        f: _safe_avg([r.get(f) for r in rows]) for f in comp_fields
    }

    return {
        "days_covered":          days_covered,
        "total_signals":         len(rows),
        "avg_per_day":           round(len(rows) / max(days_covered, 1), 1),
        "action_counts":         action_counts,
        "avg_score":             _safe_avg(all_scores),
        "median_score":          _safe_median(all_scores),
        "score_by_action":       score_by_action,
        "strategy_distribution": strategy_dist,
        "score_components_avg":  score_components_avg,
        "daily_counts":          daily_counts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Score → Return Correlation
# ─────────────────────────────────────────────────────────────────────────────

_SCORE_BUCKETS = [
    ("<40",   0,   40),
    ("40-49", 40,  50),
    ("50-59", 50,  60),
    ("60-69", 60,  70),
    ("70+",   70, 200),
]


def score_return_buckets(conn) -> list:
    """
    Answers: Does a higher score predict a better return?

    Groups resolved signal_outcomes by score range.
    Each bucket: {bucket, count, avg_t1_pnl, avg_t3_pnl, avg_t5_pnl,
                  avg_paper_pnl, paper_win_rate}
    """
    rows = conn.execute("""
        SELECT final_score, signal, direction,
               t1_pnl_pct, t3_pnl_pct, t5_pnl_pct,
               paper_pnl_pct, paper_exit, outcome
        FROM signal_outcomes
        WHERE outcome != 'PENDING'
    """).fetchall()

    if not rows:
        return []

    rows = [dict(r) for r in rows]

    result = []
    for label, lo, hi in _SCORE_BUCKETS:
        bucket = [r for r in rows
                  if r["final_score"] is not None
                  and lo <= r["final_score"] < hi]
        if not bucket:
            continue

        resolved_paper = [r for r in bucket
                          if r["paper_exit"] in ("HIT_STOP", "HIT_TARGET", "T5_EXIT")]
        paper_wins = [r for r in resolved_paper
                      if r["paper_exit"] == "HIT_TARGET"
                      or (r["paper_exit"] == "T5_EXIT"
                          and (r["paper_pnl_pct"] or 0) > 0)]

        result.append({
            "bucket":         label,
            "count":          len(bucket),
            "avg_t1_pnl":     _safe_avg([r["t1_pnl_pct"] for r in bucket]),
            "avg_t3_pnl":     _safe_avg([r["t3_pnl_pct"] for r in bucket]),
            "avg_t5_pnl":     _safe_avg([r["t5_pnl_pct"] for r in bucket]),
            "avg_paper_pnl":  _safe_avg([r["paper_pnl_pct"] for r in resolved_paper]),
            "paper_win_rate": (round(len(paper_wins) / len(resolved_paper) * 100, 1)
                               if resolved_paper else None),
        })

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. Paper Trade Summary  (Risk Engine Health)
# ─────────────────────────────────────────────────────────────────────────────

def paper_trade_summary(conn) -> dict:
    """
    Answers: Is the paper trade model and risk engine healthy?

    Returns:
      total, pending, resolved
      exit_counts / exit_pct    {HIT_STOP, HIT_TARGET, T5_EXIT}
      paper_win_rate, avg_paper_pnl, median_paper_pnl
      action_breakdown   {ACTIONABLE/WATCHLIST/MONITOR: {count, resolved,
                           avg_score, paper_win_rate, avg_paper_pnl}}
      regime_breakdown   {bull/bear/neutral/choppy: {count, avg_paper_pnl,
                           paper_win_rate}}
    """
    all_rows = conn.execute("""
        SELECT signal, regime, direction, final_score,
               paper_exit, paper_pnl_pct, outcome,
               t1_pnl_pct, t3_pnl_pct, t5_pnl_pct
        FROM signal_outcomes
    """).fetchall()

    if not all_rows:
        return {"total": 0}

    all_rows = [dict(r) for r in all_rows]
    total    = len(all_rows)
    pending  = sum(1 for r in all_rows if r["paper_exit"] in ("PENDING", None))
    resolved = [r for r in all_rows if r["paper_exit"] not in ("PENDING", None)]
    n_res    = len(resolved)

    # Exit distribution
    exit_counts = {"HIT_STOP": 0, "HIT_TARGET": 0, "T5_EXIT": 0}
    for r in resolved:
        k = r["paper_exit"]
        if k in exit_counts:
            exit_counts[k] += 1
    exit_pct = {k: round(v / n_res * 100, 1) if n_res else 0
                for k, v in exit_counts.items()}

    def _is_paper_win(r):
        return (r["paper_exit"] == "HIT_TARGET"
                or (r["paper_exit"] == "T5_EXIT"
                    and (r["paper_pnl_pct"] or 0) > 0))

    paper_wins = [r for r in resolved if _is_paper_win(r)]
    paper_pnls = [r["paper_pnl_pct"] for r in resolved if r["paper_pnl_pct"] is not None]

    # By signal level
    action_breakdown = {}
    for action in ("ACTIONABLE", "WATCHLIST", "MONITOR"):
        subset      = [r for r in all_rows if r["signal"] == action]
        res_sub     = [r for r in subset if r["paper_exit"] not in ("PENDING", None)]
        wins_sub    = [r for r in res_sub if _is_paper_win(r)]
        action_breakdown[action] = {
            "count":          len(subset),
            "resolved":       len(res_sub),
            "avg_score":      _safe_avg([r["final_score"] for r in subset]),
            "paper_win_rate": (round(len(wins_sub) / len(res_sub) * 100, 1)
                               if res_sub else None),
            "avg_paper_pnl":  _safe_avg([r["paper_pnl_pct"] for r in res_sub]),
        }

    # By regime
    regime_breakdown = {}
    for regime in ("bull", "neutral", "bear", "choppy"):
        subset   = [r for r in resolved if r["regime"] == regime]
        wins_sub = [r for r in subset if _is_paper_win(r)]
        if subset:
            regime_breakdown[regime] = {
                "count":          len(subset),
                "avg_paper_pnl":  _safe_avg([r["paper_pnl_pct"] for r in subset]),
                "paper_win_rate": round(len(wins_sub) / len(subset) * 100, 1),
            }

    return {
        "total":            total,
        "pending":          pending,
        "resolved":         n_res,
        "exit_counts":      exit_counts,
        "exit_pct":         exit_pct,
        "paper_win_rate":   (round(len(paper_wins) / n_res * 100, 1) if n_res else None),
        "avg_paper_pnl":    _safe_avg(paper_pnls),
        "median_paper_pnl": _safe_median(paper_pnls),
        "action_breakdown": action_breakdown,
        "regime_breakdown": regime_breakdown,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Event Type Breakdown
# ─────────────────────────────────────────────────────────────────────────────

_EVENT_ORDER = ["earnings", "ma", "regulation", "ai", "product", "macro", "layoff", "general"]


def event_type_breakdown(conn) -> list:
    """
    Answers: Which catalyst type actually produces returns?

    Groups resolved signal_outcomes by event_type.
    Each row: {event_type, count, avg_t5_pnl, avg_paper_pnl,
               paper_win_rate, hit_stop_pct, hit_target_pct}
    """
    rows = conn.execute("""
        SELECT event_type, paper_exit, paper_pnl_pct,
               t5_pnl_pct, outcome
        FROM signal_outcomes
        WHERE event_type IS NOT NULL AND outcome != 'PENDING'
    """).fetchall()

    if not rows:
        return []

    rows = [dict(r) for r in rows]

    # Collect all event types present, ordered by priority
    seen = {r["event_type"] for r in rows if r["event_type"]}
    ordered = [e for e in _EVENT_ORDER if e in seen]
    ordered += [e for e in seen if e not in _EVENT_ORDER]  # catch unknowns

    result = []
    for evt in ordered:
        subset = [r for r in rows if r["event_type"] == evt]
        resolved = [r for r in subset
                    if r["paper_exit"] in ("HIT_STOP", "HIT_TARGET", "T5_EXIT")]
        wins = [r for r in resolved
                if r["paper_exit"] == "HIT_TARGET"
                or (r["paper_exit"] == "T5_EXIT" and (r["paper_pnl_pct"] or 0) > 0)]
        stops  = sum(1 for r in resolved if r["paper_exit"] == "HIT_STOP")
        targets = sum(1 for r in resolved if r["paper_exit"] == "HIT_TARGET")

        n = len(resolved)
        result.append({
            "event_type":      evt,
            "count":           len(subset),
            "resolved":        n,
            "avg_t5_pnl":      _safe_avg([r["t5_pnl_pct"] for r in subset]),
            "avg_paper_pnl":   _safe_avg([r["paper_pnl_pct"] for r in resolved]),
            "paper_win_rate":  round(len(wins) / n * 100, 1) if n else None,
            "hit_stop_pct":    round(stops   / n * 100, 1) if n else None,
            "hit_target_pct":  round(targets / n * 100, 1) if n else None,
        })

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5. Weight Calibration Suggestions
# ─────────────────────────────────────────────────────────────────────────────

# Current EVENT_IMPORTANCE from analyzer.py (mirrored here to avoid circular import)
_CURRENT_WEIGHTS = {
    "earnings":   1.00,
    "macro":      0.85,
    "ma":         0.85,
    "ai":         0.80,
    "regulation": 0.75,
    "layoff":     0.60,
    "product":    0.55,
    "general":    0.20,
}

_MIN_SIGNALS_FOR_SUGGESTION = 3   # don't suggest adjustments with tiny samples


def weight_calibration_suggestions(conn) -> list:
    """
    Compares actual signal performance per event_type vs current EVENT_IMPORTANCE weights.

    Returns list of {event_type, current_weight, count, paper_win_rate,
                     avg_paper_pnl, verdict, suggestion}
    where verdict ∈ {RAISE, LOWER, OK, INSUFFICIENT_DATA}
    and suggestion is a human-readable note.
    """
    breakdown = event_type_breakdown(conn)
    if not breakdown:
        return []

    suggestions = []
    for row in breakdown:
        evt   = row["event_type"]
        n     = row["resolved"]
        wr    = row["paper_win_rate"]
        pnl   = row["avg_paper_pnl"]
        cur_w = _CURRENT_WEIGHTS.get(evt, 0.50)

        if n < _MIN_SIGNALS_FOR_SUGGESTION or wr is None:
            verdict    = "INSUFFICIENT_DATA"
            suggestion = f"Only {n} resolved signal(s) — need ≥{_MIN_SIGNALS_FOR_SUGGESTION} to calibrate."
        elif wr >= 60 and (pnl or 0) >= 1.5:
            verdict    = "RAISE"
            suggestion = (
                f"{evt}: {wr:.0f}% win rate, avg P&L {pnl:+.1f}% — "
                f"outperforming. Consider raising EVENT_IMPORTANCE['{evt}'] "
                f"from {cur_w:.2f} toward {min(cur_w + 0.10, 1.0):.2f}."
            )
        elif wr <= 40 and (pnl or 0) <= -1.0:
            verdict    = "LOWER"
            suggestion = (
                f"{evt}: {wr:.0f}% win rate, avg P&L {pnl:+.1f}% — "
                f"underperforming. Consider lowering EVENT_IMPORTANCE['{evt}'] "
                f"from {cur_w:.2f} toward {max(cur_w - 0.10, 0.05):.2f}."
            )
        else:
            verdict    = "OK"
            suggestion = (
                f"{evt}: {wr:.0f}% win rate, avg P&L {pnl:+.1f}% — "
                f"within acceptable range. Current weight {cur_w:.2f} looks reasonable."
            )

        suggestions.append({
            "event_type":     evt,
            "current_weight": cur_w,
            "count":          row["count"],
            "resolved":       n,
            "paper_win_rate": wr,
            "avg_paper_pnl":  pnl,
            "verdict":        verdict,
            "suggestion":     suggestion,
        })

    return suggestions


# ─────────────────────────────────────────────────────────────────────────────
# 6. Benchmark-Adjusted Return
# ─────────────────────────────────────────────────────────────────────────────

# Per-symbol benchmark overrides — checked before sector fallback.
# ETFs should benchmark against themselves (or the closest thematic ETF),
# not a blanket SPY.
_SYMBOL_BENCHMARK = {
    # Broad market
    "SPY":  "SPY",  "IWM": "SPY",
    # Nasdaq / tech
    "QQQ":  "QQQ",
    # Semiconductors
    "SMH":  "SMH",
    # Financials
    "XLF":  "XLF",
    # Fixed income / macro
    "TLT":  "TLT",  "GLD": "GLD",
}

# Sector → benchmark ETF ticker (must exist in price_snapshots)
_SECTOR_BENCHMARK = {
    "Technology":  "QQQ",
    "ETF":         "SPY",   # fallback for any ETF not in _SYMBOL_BENCHMARK
    "Finance":     "XLF",
    "Healthcare":  "SPY",
    "Consumer":    "SPY",
    "Energy":      "SPY",
    "Industrial":  "SPY",
    "Automotive":  "SPY",
    "Media":       "SPY",
}
_DEFAULT_BENCHMARK = "SPY"


def benchmark_adjusted_return(conn) -> dict:
    """
    Answers: Does our signal return beat a same-direction position in SPY / sector ETF?

    For each resolved signal, looks up the benchmark (SPY or sector ETF) return
    over the same holding window (signal_date → t1/t3/t5) and computes
    direction-adjusted alpha:

      LONG  signal: signal_return − benchmark_long_return
      SHORT signal: signal_return − benchmark_short_return  (benchmark return inverted)

    This is NOT "signal return vs simply holding SPY long". It answers:
    "Did we do better than a same-direction bet on the benchmark?"
    Hence the metric is called dir_adj_alpha, not plain alpha.

    Returns:
      overall        {avg_dir_adj_alpha_t1, avg_dir_adj_alpha_t3, avg_dir_adj_alpha_t5, n}
      by_sector      {sector: {avg_dir_adj_alpha_t5, avg_signal_t5, avg_bench_t5, n}}
      by_regime      {regime: {avg_dir_adj_alpha_t5, n}}
      worst          [top 5 worst dir_adj_alpha signals]
      best           [top 5 best  dir_adj_alpha signals]
    """
    signals = conn.execute("""
        SELECT so.symbol, so.signal_date, so.regime, so.direction,
               so.t1_date, so.t3_date, so.t5_date,
               so.t1_pnl_pct, so.t3_pnl_pct, so.t5_pnl_pct,
               so.final_score, so.event_type,
               ws.sector
        FROM signal_outcomes so
        LEFT JOIN watched_symbols ws ON ws.symbol = so.symbol
        WHERE so.outcome != 'PENDING'
          AND so.t5_pnl_pct IS NOT NULL
    """).fetchall()

    if not signals:
        return {}

    signals = [dict(r) for r in signals]

    # Build price lookup: {(symbol, date_str): close_price}
    price_rows = conn.execute(
        "SELECT symbol, snapshot_date, close_price FROM price_snapshots"
    ).fetchall()
    prices = {(r["symbol"], r["snapshot_date"]): r["close_price"] for r in price_rows}

    # Pre-build sorted date list per symbol for nearest-trading-day fallback.
    # price_snapshots only has trading days, so any calendar date (e.g. t3_date
    # falling on a weekend) needs to resolve to the nearest available close.
    from bisect import bisect_right
    from collections import defaultdict
    _sym_dates: dict = defaultdict(list)
    for sym, d in prices:
        _sym_dates[sym].append(d)
    for sym in _sym_dates:
        _sym_dates[sym].sort()

    def _nearest_close(sym, target_date):
        """Return the close price for sym on the nearest available trading day."""
        exact = prices.get((sym, target_date))
        if exact is not None:
            return exact
        dates = _sym_dates.get(sym)
        if not dates:
            return None
        idx = bisect_right(dates, target_date)
        # candidates: previous trading day (idx-1) and next (idx)
        candidates = []
        if idx > 0:
            candidates.append(dates[idx - 1])
        if idx < len(dates):
            candidates.append(dates[idx])
        if not candidates:
            return None
        # Pick whichever is closest in calendar days; break ties toward previous
        nearest = min(candidates, key=lambda d: (abs((date.fromisoformat(d) - date.fromisoformat(target_date)).days), d > target_date))
        return prices.get((sym, nearest))

    def _bench_return(bench_sym, start_date, end_date):
        """Benchmark % change start → end, tolerating non-trading-day dates."""
        p0 = _nearest_close(bench_sym, start_date)
        p1 = _nearest_close(bench_sym, end_date)
        if p0 and p1 and p0 > 0:
            return round((p1 - p0) / p0 * 100, 3)
        return None

    enriched = []
    for r in signals:
        sector = r.get("sector") or ""
        bench_sym = (_SYMBOL_BENCHMARK.get(r["symbol"])
                     or _SECTOR_BENCHMARK.get(sector, _DEFAULT_BENCHMARK))
        # For SHORT signals, the signal earns when price falls, but the benchmark
        # is always long — so we invert the benchmark return for fair comparison.
        direction = (r.get("direction") or "LONG").upper()
        sign = -1.0 if direction == "SHORT" else 1.0

        b1 = _bench_return(bench_sym, r["signal_date"], r["t1_date"])
        b3 = _bench_return(bench_sym, r["signal_date"], r["t3_date"])
        b5 = _bench_return(bench_sym, r["signal_date"], r["t5_date"])

        dir_adj_alpha1 = round(r["t1_pnl_pct"] - sign * b1, 3) if (b1 is not None and r["t1_pnl_pct"] is not None) else None
        dir_adj_alpha3 = round(r["t3_pnl_pct"] - sign * b3, 3) if (b3 is not None and r["t3_pnl_pct"] is not None) else None
        dir_adj_alpha5 = round(r["t5_pnl_pct"] - sign * b5, 3) if (b5 is not None and r["t5_pnl_pct"] is not None) else None

        enriched.append({**r, "benchmark": bench_sym,
                         "bench_t5": b5,
                         "dir_adj_alpha_t1": dir_adj_alpha1,
                         "dir_adj_alpha_t3": dir_adj_alpha3,
                         "dir_adj_alpha_t5": dir_adj_alpha5})

    with_alpha5 = [r for r in enriched if r["dir_adj_alpha_t5"] is not None]

    # Overall
    overall = {
        "avg_dir_adj_alpha_t1": _safe_avg([r["dir_adj_alpha_t1"] for r in enriched]),
        "avg_dir_adj_alpha_t3": _safe_avg([r["dir_adj_alpha_t3"] for r in enriched]),
        "avg_dir_adj_alpha_t5": _safe_avg([r["dir_adj_alpha_t5"] for r in with_alpha5]),
        "n":                    len(with_alpha5),
    }

    # By sector
    by_sector = {}
    for sector in set(r.get("sector") or "Unknown" for r in with_alpha5):
        sub = [r for r in with_alpha5 if (r.get("sector") or "Unknown") == sector]
        by_sector[sector] = {
            "avg_dir_adj_alpha_t5": _safe_avg([r["dir_adj_alpha_t5"] for r in sub]),
            "avg_signal_t5":        _safe_avg([r["t5_pnl_pct"] for r in sub]),
            "avg_bench_t5":         _safe_avg([r["bench_t5"] for r in sub]),
            "n":                    len(sub),
        }

    # By regime
    by_regime = {}
    for regime in ("bull", "neutral", "bear", "choppy"):
        sub = [r for r in with_alpha5 if r.get("regime") == regime]
        if sub:
            by_regime[regime] = {
                "avg_dir_adj_alpha_t5": _safe_avg([r["dir_adj_alpha_t5"] for r in sub]),
                "n":                    len(sub),
            }

    sorted_by_alpha = sorted(with_alpha5, key=lambda r: r["dir_adj_alpha_t5"] or 0)
    best  = sorted_by_alpha[-5:][::-1]
    worst = sorted_by_alpha[:5]

    def _fmt(r):
        return {
            "symbol":          r["symbol"],
            "signal_date":     r["signal_date"],
            "direction":       r["direction"],
            "t5_pnl_pct":      r["t5_pnl_pct"],
            "bench_t5":        r["bench_t5"],
            "dir_adj_alpha_t5": r["dir_adj_alpha_t5"],
            "benchmark":       r["benchmark"],
        }

    return {
        "overall":    overall,
        "by_sector":  by_sector,
        "by_regime":  by_regime,
        "best":       [_fmt(r) for r in best],
        "worst":      [_fmt(r) for r in worst],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. De-duplicated Event Return
# ─────────────────────────────────────────────────────────────────────────────

def deduplicated_event_return(conn) -> dict:
    """
    Answers: Are our returns inflated by counting the same catalyst multiple times?

    Dedup rules differ by event type — a single ISO-week key is too coarse for
    earnings (can split one report across weeks) and too fine for macro (misses
    that multiple symbols react to the same catalyst):

      earnings  → (symbol, year-quarter)
                  All post-earnings drift signals for the same ticker's quarterly
                  report collapse to one entry, regardless of week boundaries.
                  Proxy for earnings_date (not stored explicitly).

      macro     → (iso_week,)   — no symbol
                  All symbols reacting to the same macro catalyst in the same week
                  collapse to one entry.  We don't store macro_event_name in
                  signal_outcomes, so week is the finest available key.

      all other → (symbol, event_type, iso_week)   — original rule

    Within each group, keep the row with the highest final_score.

    Returns:
      raw_count          total resolved signals before dedup
      dedup_count        signals after dedup
      removed_pct        % collapsed
      raw_stats          {avg_t5_pnl, avg_paper_pnl, paper_win_rate}
      dedup_stats        same fields on deduplicated set
      by_event_type      [{event_type, raw_n, dedup_n, raw_avg_pnl, dedup_avg_pnl,
                           dedup_rule}]
    """
    rows = conn.execute("""
        SELECT symbol, signal_date, event_type, final_score,
               t5_pnl_pct, paper_pnl_pct, paper_exit, outcome
        FROM signal_outcomes
        WHERE outcome != 'PENDING'
    """).fetchall()

    if not rows:
        return {}

    rows = [dict(r) for r in rows]

    def _iso_week(date_str):
        try:
            d = date.fromisoformat(date_str)
            iso = d.isocalendar()
            return f"{iso[0]}-W{iso[1]:02d}"
        except Exception:
            return date_str[:7]

    def _year_quarter(date_str):
        try:
            d = date.fromisoformat(date_str)
            return f"{d.year}-Q{(d.month - 1) // 3 + 1}"
        except Exception:
            return date_str[:7]

    def _dedup_key(r):
        evt = r["event_type"] or "general"
        if evt == "earnings":
            return ("earnings", r["symbol"], _year_quarter(r["signal_date"]))
        if evt == "macro":
            return ("macro", _iso_week(r["signal_date"]))
        return (evt, r["symbol"], _iso_week(r["signal_date"]))

    _DEDUP_RULE = {
        "earnings": "symbol + year-quarter",
        "macro":    "iso-week only (no symbol — same catalyst, multiple tickers)",
    }

    def _is_win(r):
        return (r["paper_exit"] == "HIT_TARGET"
                or (r["paper_exit"] == "T5_EXIT" and (r["paper_pnl_pct"] or 0) > 0))

    def _stats(subset):
        resolved = [r for r in subset
                    if r["paper_exit"] in ("HIT_STOP", "HIT_TARGET", "T5_EXIT")]
        wins = [r for r in resolved if _is_win(r)]
        return {
            "avg_t5_pnl":    _safe_avg([r["t5_pnl_pct"] for r in subset]),
            "avg_paper_pnl": _safe_avg([r["paper_pnl_pct"] for r in resolved]),
            "paper_win_rate": (round(len(wins) / len(resolved) * 100, 1)
                               if resolved else None),
        }

    # De-duplicate: keep highest-score row per group
    groups: dict = {}
    for r in rows:
        key = _dedup_key(r)
        if key not in groups or (r["final_score"] or 0) > (groups[key]["final_score"] or 0):
            groups[key] = r
    dedup_rows = list(groups.values())

    # Per-event comparison
    all_event_types = sorted({r["event_type"] or "general" for r in rows})
    by_event = []
    for evt in all_event_types:
        raw_sub   = [r for r in rows       if (r["event_type"] or "general") == evt]
        dedup_sub = [r for r in dedup_rows if (r["event_type"] or "general") == evt]
        by_event.append({
            "event_type":    evt,
            "raw_n":         len(raw_sub),
            "dedup_n":       len(dedup_sub),
            "raw_avg_pnl":   _safe_avg([r["paper_pnl_pct"] for r in raw_sub]),
            "dedup_avg_pnl": _safe_avg([r["paper_pnl_pct"] for r in dedup_sub]),
            "dedup_rule":    _DEDUP_RULE.get(evt, "symbol + event_type + iso-week"),
        })

    removed = len(rows) - len(dedup_rows)
    return {
        "raw_count":    len(rows),
        "dedup_count":  len(dedup_rows),
        "removed_pct":  round(removed / len(rows) * 100, 1) if rows else 0,
        "raw_stats":    _stats(rows),
        "dedup_stats":  _stats(dedup_rows),
        "by_event_type": by_event,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. R-Multiple Analysis
# ─────────────────────────────────────────────────────────────────────────────

def r_multiple_analysis(conn) -> dict:
    """
    Answers: Are we actually capturing good R-multiples, or just grinding small wins
    while taking large losses relative to our stop?

    R  = abs(entry_price - stop_price)  — the risk unit at signal time
    R-multiple = actual_pnl_$ / R
      • HIT_TARGET should be ~+2R (model uses 2R target)
      • HIT_STOP should be ~-1R
      • T5_EXIT varies

    Returns:
      n_with_stop       signals with stop_price populated
      avg_r_multiple    overall avg R-multiple
      median_r_multiple
      expectancy        avg_r * win_rate - avg_loss_r * loss_rate  (in R)
      by_exit           {HIT_STOP, HIT_TARGET, T5_EXIT: {avg_r, median_r, n}}
      by_signal         {MONITOR/WATCHLIST/ACTIONABLE: {avg_r, n}}
      distribution      [{bucket, count}]  e.g. "<-1R", "-1R to 0", "0 to +1R", "+1R to +2R", ">+2R"
    """
    rows = conn.execute("""
        SELECT symbol, signal, direction, paper_exit, paper_pnl_pct,
               entry_price, stop_price, target_price, atr_at_signal
        FROM signal_outcomes
        WHERE paper_exit IN ('HIT_STOP', 'HIT_TARGET', 'T5_EXIT')
          AND stop_price IS NOT NULL
          AND entry_price IS NOT NULL
          AND entry_price > 0
    """).fetchall()

    if not rows:
        return {"n_with_stop": 0}

    rows = [dict(r) for r in rows]

    def _r_multiple(r):
        risk = abs(r["entry_price"] - r["stop_price"])
        if risk <= 0:
            return None
        # paper_pnl_pct is already directional (positive = profitable)
        pnl_dollar = (r["paper_pnl_pct"] / 100.0) * r["entry_price"]
        return round(pnl_dollar / risk, 3)

    for r in rows:
        r["r_mult"] = _r_multiple(r)

    valid = [r for r in rows if r["r_mult"] is not None]
    r_vals = [r["r_mult"] for r in valid]

    # Expectancy in R: E = avg_win_R * win_rate - avg_loss_R * loss_rate
    wins   = [r["r_mult"] for r in valid if r["r_mult"] > 0]
    losses = [r["r_mult"] for r in valid if r["r_mult"] <= 0]
    n = len(valid)
    expectancy = None
    if n > 0:
        wr = len(wins) / n
        lr = len(losses) / n
        avg_win_r  = sum(wins)   / len(wins)   if wins   else 0
        avg_loss_r = sum(losses) / len(losses) if losses else 0
        expectancy = round(wr * avg_win_r + lr * avg_loss_r, 3)

    # By exit type
    by_exit = {}
    for exit_type in ("HIT_STOP", "HIT_TARGET", "T5_EXIT"):
        sub = [r["r_mult"] for r in valid if r["paper_exit"] == exit_type]
        if sub:
            by_exit[exit_type] = {
                "avg_r":    round(sum(sub) / len(sub), 3),
                "median_r": round(statistics.median(sub), 3),
                "n":        len(sub),
            }

    # By signal level
    by_signal = {}
    for sig in ("MONITOR", "WATCHLIST", "ACTIONABLE"):
        sub = [r["r_mult"] for r in valid if r["signal"] == sig]
        if sub:
            by_signal[sig] = {
                "avg_r": round(sum(sub) / len(sub), 3),
                "n":     len(sub),
            }

    # Distribution buckets
    dist_buckets = [
        ("< -1R",     lambda v: v < -1),
        ("-1R to 0",  lambda v: -1 <= v < 0),
        ("0 to +1R",  lambda v: 0 <= v < 1),
        ("+1R to +2R",lambda v: 1 <= v < 2),
        ("> +2R",     lambda v: v >= 2),
    ]
    distribution = [
        {"bucket": label, "count": sum(1 for v in r_vals if fn(v))}
        for label, fn in dist_buckets
    ]

    return {
        "n_with_stop":       len(valid),
        "avg_r_multiple":    round(sum(r_vals) / len(r_vals), 3) if r_vals else None,
        "median_r_multiple": round(statistics.median(r_vals), 3) if r_vals else None,
        "expectancy":        expectancy,
        "by_exit":           by_exit,
        "by_signal":         by_signal,
        "distribution":      distribution,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 9. Win-Rate Confidence Intervals  (Wilson score, 95%)
# ─────────────────────────────────────────────────────────────────────────────

def _wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score confidence interval. Returns (lower, upper) as percentages."""
    if n == 0:
        return (None, None)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half   = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round((centre - half) * 100, 1), round((centre + half) * 100, 1))


def win_rate_confidence_intervals(conn) -> list:
    """
    Answers: How reliable is the win rate estimate for each event type?
    A 70% win rate on 5 signals is noise; on 80 signals it's signal.

    Returns list of rows sorted by event_type, each with:
      event_type, n, wins, win_rate_pct, ci_low, ci_high, ci_width, reliable
    where `reliable` = True if the CI width is ≤ 20 percentage points.
    """
    rows = conn.execute("""
        SELECT event_type, paper_exit, paper_pnl_pct
        FROM signal_outcomes
        WHERE paper_exit IN ('HIT_STOP', 'HIT_TARGET', 'T5_EXIT')
    """).fetchall()

    if not rows:
        return []

    rows = [dict(r) for r in rows]

    def _is_win(r):
        return (r["paper_exit"] == "HIT_TARGET"
                or (r["paper_exit"] == "T5_EXIT" and (r["paper_pnl_pct"] or 0) > 0))

    # Overall row first
    results = []
    wins_all = sum(1 for r in rows if _is_win(r))
    n_all    = len(rows)
    ci_lo, ci_hi = _wilson_ci(wins_all, n_all)
    results.append({
        "event_type":   "ALL",
        "n":            n_all,
        "wins":         wins_all,
        "win_rate_pct": round(wins_all / n_all * 100, 1) if n_all else None,
        "ci_low":       ci_lo,
        "ci_high":      ci_hi,
        "ci_width":     round(ci_hi - ci_lo, 1) if (ci_lo is not None and ci_hi is not None) else None,
        "reliable":     (ci_hi - ci_lo) <= 20 if (ci_lo is not None and ci_hi is not None) else False,
    })

    # Per event type
    all_types = sorted({r["event_type"] or "general" for r in rows})
    for evt in all_types:
        sub  = [r for r in rows if (r["event_type"] or "general") == evt]
        n    = len(sub)
        wins = sum(1 for r in sub if _is_win(r))
        ci_lo, ci_hi = _wilson_ci(wins, n)
        width = round(ci_hi - ci_lo, 1) if (ci_lo is not None and ci_hi is not None) else None
        results.append({
            "event_type":   evt,
            "n":            n,
            "wins":         wins,
            "win_rate_pct": round(wins / n * 100, 1) if n else None,
            "ci_low":       ci_lo,
            "ci_high":      ci_hi,
            "ci_width":     width,
            "reliable":     width <= 20 if width is not None else False,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 10. Component Correlation Report
# ─────────────────────────────────────────────────────────────────────────────

_COMPONENTS = [
    ("event_edge_score",   "EventEdge"),
    ("market_conf_score",  "MarketConf"),
    ("regime_fit_score",   "RegimeFit"),
    ("relative_opp_score", "RelativeOpp"),
    ("freshness_score",    "Freshness"),
    ("risk_penalty_score", "RiskPenalty"),
]

_MIN_CORR_N = 10  # suppress correlation rows with fewer valid pairs


def _pearson(xs, ys):
    """Pearson correlation. Returns None when fewer than _MIN_CORR_N valid pairs."""
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    n = len(pairs)
    if n < _MIN_CORR_N:
        return None
    xs_ = [p[0] for p in pairs]
    ys_ = [p[1] for p in pairs]
    mx = sum(xs_) / n
    my = sum(ys_) / n
    num   = sum((x - mx) * (y - my) for x, y in zip(xs_, ys_))
    denom = (sum((x - mx) ** 2 for x in xs_) * sum((y - my) ** 2 for y in ys_)) ** 0.5
    if denom == 0:
        return None
    return round(num / denom, 3)


def component_correlation_report(conn) -> dict:
    """
    Answers: Which score components actually predict returns, and which are noise?

    Joins signal_outcomes (outcomes) with trade_candidates (per-component scores)
    on symbol + date, then computes Pearson r for each component vs three outcomes:

      corr_t5     — raw t+5 return
      corr_alpha  — direction-adjusted alpha vs sector benchmark (same logic as
                    benchmark_adjusted_return; benchmark lookup uses nearest trading day)
      corr_r      — R-multiple  (paper_pnl / stop-risk; None when stop_price missing)

    Note on RiskPenalty sign: it is *subtracted* in the final score formula.
    A positive corr_t5 for RiskPenalty means high-penalty signals still returned
    well; a negative corr means the penalty correctly identified losing trades.

    Returns:
      overall      [{component, label, n, corr_t5, corr_alpha, corr_r}]
      by_bucket    {strategy_bucket: [{component, label, n, corr_t5, corr_alpha, corr_r}]}
                   only buckets with ≥ _MIN_CORR_N signals are included
    """
    rows = conn.execute("""
        SELECT so.symbol, so.signal_date, so.t5_date, so.direction,
               so.t5_pnl_pct, so.paper_pnl_pct, so.paper_exit,
               so.entry_price, so.stop_price, so.strategy_bucket,
               ws.sector,
               tc.event_edge_score, tc.market_conf_score, tc.regime_fit_score,
               tc.relative_opp_score, tc.freshness_score, tc.risk_penalty_score
        FROM signal_outcomes so
        JOIN trade_candidates tc
          ON tc.symbol = so.symbol AND tc.run_date = so.signal_date
        LEFT JOIN watched_symbols ws ON ws.symbol = so.symbol
        WHERE so.outcome != 'PENDING'
          AND so.t5_pnl_pct IS NOT NULL
    """).fetchall()

    if not rows:
        return {}

    rows = [dict(r) for r in rows]

    # ── R-multiple ────────────────────────────────────────────────────────────
    def _r_mult(r):
        ep  = r.get("entry_price")
        sp  = r.get("stop_price")
        pnl = r.get("paper_pnl_pct")
        if ep and sp and ep > 0 and pnl is not None:
            risk = abs(ep - sp)
            if risk > 0:
                return round((pnl / 100.0) * ep / risk, 3)
        return None

    for r in rows:
        r["r_multiple"] = _r_mult(r)

    # ── Direction-adjusted alpha (nearest-trading-day benchmark lookup) ───────
    from bisect import bisect_right
    from collections import defaultdict

    price_rows = conn.execute(
        "SELECT symbol, snapshot_date, close_price FROM price_snapshots"
    ).fetchall()
    _prices = {(pr["symbol"], pr["snapshot_date"]): pr["close_price"] for pr in price_rows}

    _sym_dates_c: dict = defaultdict(list)
    for sym, d in _prices:
        _sym_dates_c[sym].append(d)
    for sym in _sym_dates_c:
        _sym_dates_c[sym].sort()

    def _nc(sym, target):
        exact = _prices.get((sym, target))
        if exact is not None:
            return exact
        dates = _sym_dates_c.get(sym)
        if not dates:
            return None
        idx = bisect_right(dates, target)
        candidates = []
        if idx > 0:         candidates.append(dates[idx - 1])
        if idx < len(dates): candidates.append(dates[idx])
        if not candidates:
            return None
        nearest = min(candidates, key=lambda d: (
            abs((date.fromisoformat(d) - date.fromisoformat(target)).days),
            d > target,
        ))
        return _prices.get((sym, nearest))

    for r in rows:
        sector    = r.get("sector") or ""
        bench_sym = (_SYMBOL_BENCHMARK.get(r["symbol"])
                     or _SECTOR_BENCHMARK.get(sector, _DEFAULT_BENCHMARK))
        sign      = -1.0 if (r.get("direction") or "LONG").upper() == "SHORT" else 1.0
        p0 = _nc(bench_sym, r["signal_date"])
        p1 = _nc(bench_sym, r["t5_date"]) if r.get("t5_date") else None
        if p0 and p1 and p0 > 0:
            bench_ret = (p1 - p0) / p0 * 100
            r["dir_adj_alpha"] = round(r["t5_pnl_pct"] - sign * bench_ret, 3)
        else:
            r["dir_adj_alpha"] = None

    # ── Correlation helper ────────────────────────────────────────────────────
    def _corr_rows(subset):
        t5    = [r["t5_pnl_pct"]    for r in subset]
        alpha = [r["dir_adj_alpha"] for r in subset]
        rmult = [r["r_multiple"]    for r in subset]
        result = []
        for field, label in _COMPONENTS:
            xs = [r[field] for r in subset]
            n_valid = sum(1 for x in xs if x is not None)
            result.append({
                "component":  label,
                "n":          n_valid,
                "corr_t5":    _pearson(xs, t5),
                "corr_alpha": _pearson(xs, alpha),
                "corr_r":     _pearson(xs, rmult),
            })
        return result

    # ── Overall ───────────────────────────────────────────────────────────────
    overall = _corr_rows(rows)

    # ── By strategy bucket (suppress buckets with < _MIN_CORR_N rows) ─────────
    by_bucket = {}
    all_buckets = sorted({r["strategy_bucket"] or "unknown" for r in rows})
    for bucket in all_buckets:
        sub = [r for r in rows if (r["strategy_bucket"] or "unknown") == bucket]
        if len(sub) >= _MIN_CORR_N:
            by_bucket[bucket] = _corr_rows(sub)

    return {
        "overall":   overall,
        "by_bucket": by_bucket,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 11. Before / After Rescore Comparison
# ─────────────────────────────────────────────────────────────────────────────

def rescore_comparison_report(conn) -> dict:
    """
    Re-applies new risk scoring logic to historical signals and compares
    old vs new scores.  Outcomes (t5_pnl, paper_pnl, paper_exit) are unchanged.

    What changed in the new scoring:
      - score_risk_penalty: choppy penalty removed; RSI now event-type aware
      - score_post_earnings_risk: earnings signals get strength_bonus on EventEdge

    Approximation: news_scores.mixedness is set to 0 (cannot reconstruct from DB).
    Mixed-news penalties are therefore underestimated for both old and new; the
    relative old-vs-new comparison is still valid for choppy/RSI-driven changes.

    new_score formula:
      new_score = old_score + earn_strength + (old_rp - new_rp)
    where earn_strength = 0 for non-earnings signals.

    Returns:
      n                 signals rescored
      summary           {old_avg_score, new_avg_score, old_avg_rp, new_avg_rp,
                         avg_score_delta}
      tier_changes      [{old_tier, new_tier, count}]  sorted by count desc
      score_buckets     [{bucket, old_n, new_n, old_win_rate, new_win_rate,
                         old_avg_pnl, new_avg_pnl}]
      correlation       {old_corr_t5, new_corr_t5,
                         old_corr_alpha, new_corr_alpha,
                         old_corr_r, new_corr_r}
      biggest_upgrades  [top 10 largest score increases]
      biggest_downgrades [top 10 largest score decreases]
    """
    # Local import — avoids module-level circular dependency
    from modules.analyzer import (
        score_risk_penalty, score_post_earnings_risk,
        determine_action,
    )
    from bisect import bisect_right
    from collections import defaultdict

    rows = conn.execute("""
        SELECT so.symbol, so.signal_date, so.t5_date, so.direction, so.regime,
               so.event_type, so.t5_pnl_pct, so.paper_pnl_pct, so.paper_exit,
               so.entry_price, so.stop_price,
               tc.final_score      AS old_score,
               tc.event_edge_score AS old_ee,
               tc.risk_penalty_score AS old_rp,
               ps.change_pct, ps.rsi_14, ps.atr_14, ps.close_price,
               ps.volume_ratio, ps.day_high, ps.day_low,
               ws.sector
        FROM signal_outcomes so
        JOIN trade_candidates tc
          ON tc.symbol = so.symbol AND tc.run_date = so.signal_date
        JOIN price_snapshots ps
          ON ps.symbol = so.symbol AND ps.snapshot_date = so.signal_date
        LEFT JOIN watched_symbols ws ON ws.symbol = so.symbol
        WHERE so.outcome != 'PENDING'
          AND so.t5_pnl_pct IS NOT NULL
    """).fetchall()

    if not rows:
        return {}

    rows = [dict(r) for r in rows]

    # ── Rescore each row ──────────────────────────────────────────────────────
    for r in rows:
        price_row = {k: r[k] for k in (
            "change_pct", "rsi_14", "atr_14", "close_price",
            "volume_ratio", "day_high", "day_low",
        )}
        # Reconstruct minimal news_scores — mixedness unknown, defaults to 0
        news_scores = {
            "best_event_type": r["event_type"] or "general",
            "mixedness": 0.0,
        }
        regime_dict = {"regime": r["regime"] or "neutral"}
        direction   = r["direction"] or "LONG"

        old_rp    = r["old_rp"]    or 0.0
        old_score = r["old_score"] or 0.0

        if (r["event_type"] or "") == "earnings":
            earn_strength, new_rp = score_post_earnings_risk(
                price_row, direction, news_scores, regime_dict
            )
        else:
            earn_strength = 0.0
            new_rp = score_risk_penalty(
                price_row, direction, news_scores, regime_dict
            )

        # new_score = old_score + earn_strength + (old_rp - new_rp)
        new_score = old_score + earn_strength + (old_rp - new_rp)

        r["new_score"]        = round(new_score, 2)
        r["new_rp"]           = round(new_rp, 2)
        r["earn_strength"]    = round(earn_strength, 1)
        r["score_delta"]      = round(new_score - old_score, 2)
        r["rp_delta"]         = round(new_rp - old_rp, 2)
        r["old_tier"]         = determine_action(old_score, direction, regime_dict)
        r["new_tier"]         = determine_action(new_score, direction, regime_dict)

    # ── R-multiple ────────────────────────────────────────────────────────────
    def _r_mult(r):
        ep  = r.get("entry_price")
        sp  = r.get("stop_price")
        pnl = r.get("paper_pnl_pct")
        if ep and sp and ep > 0 and pnl is not None:
            risk = abs(ep - sp)
            if risk > 0:
                return round((pnl / 100.0) * ep / risk, 3)
        return None

    for r in rows:
        r["r_multiple"] = _r_mult(r)

    # ── Direction-adjusted alpha (nearest-trading-day benchmark lookup) ───────
    price_rows = conn.execute(
        "SELECT symbol, snapshot_date, close_price FROM price_snapshots"
    ).fetchall()
    _prices = {(pr["symbol"], pr["snapshot_date"]): pr["close_price"]
               for pr in price_rows}

    _sym_dates_r: dict = defaultdict(list)
    for sym, d in _prices:
        _sym_dates_r[sym].append(d)
    for sym in _sym_dates_r:
        _sym_dates_r[sym].sort()

    def _nc(sym, target):
        exact = _prices.get((sym, target))
        if exact is not None:
            return exact
        dates = _sym_dates_r.get(sym)
        if not dates:
            return None
        idx = bisect_right(dates, target)
        candidates = []
        if idx > 0:          candidates.append(dates[idx - 1])
        if idx < len(dates): candidates.append(dates[idx])
        if not candidates:
            return None
        nearest = min(candidates, key=lambda d: (
            abs((date.fromisoformat(d) - date.fromisoformat(target)).days),
            d > target,
        ))
        return _prices.get((sym, nearest))

    for r in rows:
        sector    = r.get("sector") or ""
        bench_sym = (_SYMBOL_BENCHMARK.get(r["symbol"])
                     or _SECTOR_BENCHMARK.get(sector, _DEFAULT_BENCHMARK))
        sign      = -1.0 if (r.get("direction") or "LONG").upper() == "SHORT" else 1.0
        p0 = _nc(bench_sym, r["signal_date"])
        p1 = _nc(bench_sym, r["t5_date"]) if r.get("t5_date") else None
        if p0 and p1 and p0 > 0:
            bench_ret = (p1 - p0) / p0 * 100
            r["dir_adj_alpha"] = round(r["t5_pnl_pct"] - sign * bench_ret, 3)
        else:
            r["dir_adj_alpha"] = None

    # ── Summary stats ─────────────────────────────────────────────────────────
    n = len(rows)
    summary = {
        "n":               n,
        "old_avg_score":   _safe_avg([r["old_score"] for r in rows]),
        "new_avg_score":   _safe_avg([r["new_score"] for r in rows]),
        "old_avg_rp":      _safe_avg([r["old_rp"]    for r in rows]),
        "new_avg_rp":      _safe_avg([r["new_rp"]    for r in rows]),
        "avg_score_delta": _safe_avg([r["score_delta"] for r in rows]),
        "pct_upgraded":    round(sum(1 for r in rows if r["score_delta"] > 0.5) / n * 100, 1),
        "pct_downgraded":  round(sum(1 for r in rows if r["score_delta"] < -0.5) / n * 100, 1),
    }

    # ── Tier changes ─────────────────────────────────────────────────────────
    tier_map: dict = {}
    for r in rows:
        key = (r["old_tier"], r["new_tier"])
        tier_map[key] = tier_map.get(key, 0) + 1
    tier_changes = [
        {"old_tier": k[0], "new_tier": k[1], "count": v}
        for k, v in sorted(tier_map.items(), key=lambda x: -x[1])
    ]

    # ── Score buckets: old and new distributions ──────────────────────────────
    def _bucket_label(score):
        if score is None:   return None
        if score < 40:      return "<40"
        if score < 50:      return "40-49"
        if score < 60:      return "50-59"
        if score < 70:      return "60-69"
        return "70+"

    def _is_win(r):
        return (r["paper_exit"] == "HIT_TARGET"
                or (r["paper_exit"] == "T5_EXIT" and (r["paper_pnl_pct"] or 0) > 0))

    bucket_labels = ["<40", "40-49", "50-59", "60-69", "70+"]
    buckets_out = []
    for label in bucket_labels:
        old_sub = [r for r in rows if _bucket_label(r["old_score"]) == label]
        new_sub = [r for r in rows if _bucket_label(r["new_score"]) == label]
        if not old_sub and not new_sub:
            continue
        old_wins = [r for r in old_sub if _is_win(r)]
        new_wins = [r for r in new_sub if _is_win(r)]
        buckets_out.append({
            "bucket":       label,
            "old_n":        len(old_sub),
            "new_n":        len(new_sub),
            "old_win_rate": round(len(old_wins) / len(old_sub) * 100, 1) if old_sub else None,
            "new_win_rate": round(len(new_wins) / len(new_sub) * 100, 1) if new_sub else None,
            "old_avg_pnl":  _safe_avg([r["paper_pnl_pct"] for r in old_sub]),
            "new_avg_pnl":  _safe_avg([r["paper_pnl_pct"] for r in new_sub]),
        })

    # ── Correlation: old_score and new_score vs outcomes ─────────────────────
    old_scores = [r["old_score"]  for r in rows]
    new_scores = [r["new_score"]  for r in rows]
    t5s        = [r["t5_pnl_pct"] for r in rows]
    alphas     = [r["dir_adj_alpha"] for r in rows]
    rmults     = [r["r_multiple"]    for r in rows]

    correlation = {
        "old_corr_t5":    _pearson(old_scores, t5s),
        "new_corr_t5":    _pearson(new_scores, t5s),
        "old_corr_alpha": _pearson(old_scores, alphas),
        "new_corr_alpha": _pearson(new_scores, alphas),
        "old_corr_r":     _pearson(old_scores, rmults),
        "new_corr_r":     _pearson(new_scores, rmults),
    }

    # ── Biggest movers ────────────────────────────────────────────────────────
    def _fmt_mover(r):
        return {
            "symbol":      r["symbol"],
            "signal_date": r["signal_date"],
            "event_type":  r["event_type"] or "—",
            "direction":   r["direction"],
            "regime":      r["regime"],
            "old_score":   r["old_score"],
            "new_score":   r["new_score"],
            "delta":       r["score_delta"],
            "earn_str":    r["earn_strength"],
            "old_rp":      r["old_rp"],
            "new_rp":      r["new_rp"],
            "t5_pnl":      r["t5_pnl_pct"],
        }

    sorted_delta = sorted(rows, key=lambda r: r["score_delta"])
    upgrades   = [_fmt_mover(r) for r in sorted_delta[-10:][::-1]]
    downgrades = [_fmt_mover(r) for r in sorted_delta[:10]]

    return {
        "summary":           summary,
        "tier_changes":      tier_changes,
        "score_buckets":     buckets_out,
        "correlation":       correlation,
        "biggest_upgrades":  upgrades,
        "biggest_downgrades": downgrades,
    }


def promoted_signal_quality_report(conn) -> dict:
    """
    For signals that gained score in the rescore, how do they actually perform?

    Groups by promotion criteria:
      - score_increase >= 1
      - score_increase >= 3
      - score_increase >= 5
      - old <50 and new >=50   (crossed from MONITOR-zone into WATCHLIST-zone)
      - old MONITOR -> new WATCHLIST  (explicit tier upgrade)

    Each group outputs:
      n, win_rate, avg_t5_pnl, avg_alpha_t5, avg_r_multiple,
      hit_target_rate, hit_stop_rate, t5_exit_rate

    Also includes a "baseline" group (all resolved signals, no filter) for comparison.
    """
    from modules.analyzer import (
        score_risk_penalty, score_post_earnings_risk,
        determine_action,
    )
    from bisect import bisect_right
    from collections import defaultdict

    rows = conn.execute("""
        SELECT so.symbol, so.signal_date, so.t5_date, so.direction, so.regime,
               so.event_type, so.t5_pnl_pct, so.paper_pnl_pct, so.paper_exit,
               so.entry_price, so.stop_price,
               tc.final_score      AS old_score,
               tc.risk_penalty_score AS old_rp,
               ps.change_pct, ps.rsi_14, ps.atr_14, ps.close_price,
               ps.volume_ratio, ps.day_high, ps.day_low,
               ws.sector
        FROM signal_outcomes so
        JOIN trade_candidates tc
          ON tc.symbol = so.symbol AND tc.run_date = so.signal_date
        JOIN price_snapshots ps
          ON ps.symbol = so.symbol AND ps.snapshot_date = so.signal_date
        LEFT JOIN watched_symbols ws ON ws.symbol = so.symbol
        WHERE so.outcome != 'PENDING'
          AND so.t5_pnl_pct IS NOT NULL
    """).fetchall()

    if not rows:
        return {}

    rows = [dict(r) for r in rows]

    # ── Rescore (same logic as rescore_comparison_report) ────────────────────
    for r in rows:
        price_row = {k: r[k] for k in (
            "change_pct", "rsi_14", "atr_14", "close_price",
            "volume_ratio", "day_high", "day_low",
        )}
        news_scores = {
            "best_event_type": r["event_type"] or "general",
            "mixedness": 0.0,
        }
        regime_dict = {"regime": r["regime"] or "neutral"}
        direction   = r["direction"] or "LONG"
        old_rp      = r["old_rp"]    or 0.0
        old_score   = r["old_score"] or 0.0

        if (r["event_type"] or "") == "earnings":
            earn_strength, new_rp = score_post_earnings_risk(
                price_row, direction, news_scores, regime_dict
            )
        else:
            earn_strength = 0.0
            new_rp = score_risk_penalty(
                price_row, direction, news_scores, regime_dict
            )

        new_score = old_score + earn_strength + (old_rp - new_rp)
        r["new_score"]  = round(new_score, 2)
        r["new_rp"]     = round(new_rp, 2)
        r["score_delta"] = round(new_score - old_score, 2)
        r["old_tier"]   = determine_action(old_score, direction, regime_dict)
        r["new_tier"]   = determine_action(new_score, direction, regime_dict)

    # ── R-multiple ────────────────────────────────────────────────────────────
    def _r_mult(r):
        ep  = r.get("entry_price")
        sp  = r.get("stop_price")
        pnl = r.get("paper_pnl_pct")
        if ep and sp and ep > 0 and pnl is not None:
            risk = abs(ep - sp)
            if risk > 0:
                return round((pnl / 100.0) * ep / risk, 3)
        return None

    for r in rows:
        r["r_multiple"] = _r_mult(r)

    # ── Direction-adjusted alpha ──────────────────────────────────────────────
    price_rows = conn.execute(
        "SELECT symbol, snapshot_date, close_price FROM price_snapshots"
    ).fetchall()
    _prices = {(pr["symbol"], pr["snapshot_date"]): pr["close_price"]
               for pr in price_rows}
    _sym_dates_r: dict = defaultdict(list)
    for sym, d in _prices:
        _sym_dates_r[sym].append(d)
    for sym in _sym_dates_r:
        _sym_dates_r[sym].sort()

    def _nc(sym, target):
        exact = _prices.get((sym, target))
        if exact is not None:
            return exact
        dates = _sym_dates_r.get(sym)
        if not dates:
            return None
        from datetime import date as _date
        idx = bisect_right(dates, target)
        candidates = []
        if idx > 0:          candidates.append(dates[idx - 1])
        if idx < len(dates): candidates.append(dates[idx])
        if not candidates:
            return None
        nearest = min(candidates, key=lambda d: (
            abs((_date.fromisoformat(d) - _date.fromisoformat(target)).days),
            d > target,
        ))
        return _prices.get((sym, nearest))

    for r in rows:
        sector    = r.get("sector") or ""
        bench_sym = (_SYMBOL_BENCHMARK.get(r["symbol"])
                     or _SECTOR_BENCHMARK.get(sector, _DEFAULT_BENCHMARK))
        sign      = -1.0 if (r.get("direction") or "LONG").upper() == "SHORT" else 1.0
        p0 = _nc(bench_sym, r["signal_date"])
        p1 = _nc(bench_sym, r["t5_date"]) if r.get("t5_date") else None
        if p0 and p1 and p0 > 0:
            bench_ret = (p1 - p0) / p0 * 100
            r["dir_adj_alpha"] = round(r["t5_pnl_pct"] - sign * bench_ret, 3)
        else:
            r["dir_adj_alpha"] = None

    # ── Group stats helper ────────────────────────────────────────────────────
    def _group_stats(subset):
        n = len(subset)
        if n == 0:
            return {"n": 0, "win_rate": None, "avg_t5_pnl": None,
                    "avg_alpha_t5": None, "avg_r_multiple": None,
                    "hit_target_rate": None, "hit_stop_rate": None,
                    "t5_exit_rate": None}

        def _is_win(r):
            return (r["paper_exit"] == "HIT_TARGET"
                    or (r["paper_exit"] == "T5_EXIT" and (r["paper_pnl_pct"] or 0) > 0))

        wins = sum(1 for r in subset if _is_win(r))
        hit_target = sum(1 for r in subset if r["paper_exit"] == "HIT_TARGET")
        hit_stop   = sum(1 for r in subset if r["paper_exit"] == "HIT_STOP")
        t5_exit    = sum(1 for r in subset if r["paper_exit"] == "T5_EXIT")

        t5_pnls  = [r["t5_pnl_pct"]    for r in subset if r["t5_pnl_pct"]    is not None]
        alphas   = [r["dir_adj_alpha"]  for r in subset if r["dir_adj_alpha"] is not None]
        rmults   = [r["r_multiple"]     for r in subset if r["r_multiple"]    is not None]

        return {
            "n":              n,
            "win_rate":       round(wins / n * 100, 1),
            "avg_t5_pnl":    _safe_avg(t5_pnls),
            "avg_alpha_t5":  _safe_avg(alphas),
            "avg_r_multiple": _safe_avg(rmults),
            "hit_target_rate": round(hit_target / n * 100, 1),
            "hit_stop_rate":   round(hit_stop   / n * 100, 1),
            "t5_exit_rate":    round(t5_exit    / n * 100, 1),
        }

    # ── Build groups ──────────────────────────────────────────────────────────
    groups = {
        "baseline (all)":          rows,
        "Δscore ≥ 1":              [r for r in rows if r["score_delta"] >= 1],
        "Δscore ≥ 3":              [r for r in rows if r["score_delta"] >= 3],
        "Δscore ≥ 5":              [r for r in rows if r["score_delta"] >= 5],
        "old <50 → new ≥50":       [r for r in rows
                                    if r["old_score"] < 50 and r["new_score"] >= 50],
        "MONITOR → WATCHLIST":     [r for r in rows
                                    if r["old_tier"] == "MONITOR"
                                    and r["new_tier"] == "WATCHLIST"],
    }

    result = []
    for label, subset in groups.items():
        stats = _group_stats(subset)
        stats["group"] = label
        result.append(stats)

    return {"groups": result}


def false_upgrade_diagnosis_report(conn) -> dict:
    """
    Pulls all signals where rescore delta >= 3 but t5_pnl_pct < 0 (false upgrades).
    Returns per-signal diagnostic rows plus pattern summary across the group.

    Per-signal fields:
      symbol, signal_date, direction, event_type, strategy_bucket, regime,
      gap_pct, rsi, atr_pct, volume_ratio,
      market_conf, relative_opp, freshness,
      old_score, new_score, score_delta,
      old_risk_penalty, new_risk_penalty, rp_delta,
      earn_strength,
      t5_pnl, paper_exit, paper_pnl,
      reason_for_upgrade   ← human-readable explanation of why score rose

    Pattern summary (over the false-upgrade group):
      n, common_event_types, common_directions, common_regimes,
      avg_gap_pct, avg_rsi, avg_atr_pct, avg_volume_ratio,
      pct_earnings, pct_short, pct_choppy, pct_large_gap
    """
    from modules.analyzer import (
        score_risk_penalty, score_post_earnings_risk,
        determine_action,
    )

    rows = conn.execute("""
        SELECT so.symbol, so.signal_date, so.direction, so.regime,
               so.event_type, so.t5_pnl_pct, so.paper_pnl_pct, so.paper_exit,
               so.entry_price, so.stop_price,
               tc.final_score          AS old_score,
               tc.risk_penalty_score   AS old_rp,
               tc.market_conf_score    AS market_conf,
               tc.relative_opp_score   AS relative_opp,
               tc.freshness_score      AS freshness,
               tc.strategy_bucket,
               ps.change_pct, ps.rsi_14, ps.atr_14, ps.close_price,
               ps.volume_ratio, ps.day_high, ps.day_low
        FROM signal_outcomes so
        JOIN trade_candidates tc
          ON tc.symbol = so.symbol AND tc.run_date = so.signal_date
        JOIN price_snapshots ps
          ON ps.symbol = so.symbol AND ps.snapshot_date = so.signal_date
        WHERE so.outcome != 'PENDING'
          AND so.t5_pnl_pct IS NOT NULL
    """).fetchall()

    if not rows:
        return {}

    rows = [dict(r) for r in rows]

    # ── Rescore ───────────────────────────────────────────────────────────────
    for r in rows:
        price_row = {k: r[k] for k in (
            "change_pct", "rsi_14", "atr_14", "close_price",
            "volume_ratio", "day_high", "day_low",
        )}
        news_scores = {
            "best_event_type": r["event_type"] or "general",
            "mixedness": 0.0,
        }
        regime_dict = {"regime": r["regime"] or "neutral"}
        direction   = r["direction"] or "LONG"
        old_rp      = r["old_rp"]    or 0.0
        old_score   = r["old_score"] or 0.0

        if (r["event_type"] or "") == "earnings":
            earn_strength, new_rp = score_post_earnings_risk(
                price_row, direction, news_scores, regime_dict
            )
        else:
            earn_strength = 0.0
            new_rp = score_risk_penalty(
                price_row, direction, news_scores, regime_dict
            )

        new_score = old_score + earn_strength + (old_rp - new_rp)
        r["new_score"]    = round(new_score, 2)
        r["new_rp"]       = round(new_rp, 2)
        r["score_delta"]  = round(new_score - old_score, 2)
        r["earn_strength"] = round(earn_strength, 1)
        r["rp_delta"]     = round(new_rp - old_rp, 2)

    # ── Filter: promoted (Δ≥3) AND loser (t5<0) ──────────────────────────────
    false_ups = [r for r in rows if r["score_delta"] >= 3 and r["t5_pnl_pct"] < 0]

    if not false_ups:
        return {"signals": [], "pattern": {"n": 0}, "promoted_n": 0,
                "all_promoted_n": sum(1 for r in rows if r["score_delta"] >= 3)}

    # ── Build reason_for_upgrade ──────────────────────────────────────────────
    def _reason(r):
        parts = []
        rp_saved  = round(r["old_rp"] - r["new_rp"], 2)
        earn_str  = r["earn_strength"]
        rsi       = r.get("rsi_14") or 0
        atr_pct   = round((r.get("atr_14") or 0) / (r.get("close_price") or 1) * 100, 1)
        regime    = r.get("regime") or ""
        direction = r.get("direction") or ""
        evt       = r.get("event_type") or "general"

        if earn_str > 0:
            parts.append(f"earnings gap bonus +{earn_str:.1f}")
        if rp_saved > 0:
            parts.append(f"RP reduced by {rp_saved:.1f}")
            # Explain what penalty was removed
            if rsi > 70 and direction == "SHORT":
                parts.append(f"(RSI {rsi:.0f} no longer penalised for SHORT)")
            if rsi > 75 and direction == "LONG" and evt not in ("general", "macro"):
                parts.append(f"(RSI {rsi:.0f} no longer penalised for event-driven LONG)")
            if "choppy" in regime:
                parts.append("(choppy penalty removed)")
        if not parts:
            parts.append(f"score_delta={r['score_delta']:+.1f} (other)")
        return "; ".join(parts)

    # ── Format output rows ────────────────────────────────────────────────────
    signal_rows = []
    for r in sorted(false_ups, key=lambda x: x["score_delta"], reverse=True):
        close  = r.get("close_price") or 0
        atr    = r.get("atr_14")      or 0
        dhi    = r.get("day_high")
        dlo    = r.get("day_low")
        # gap = open-vs-prior-close proxy: use change_pct as best available
        gap_pct = round(r.get("change_pct") or 0, 2)

        signal_rows.append({
            "symbol":            r["symbol"],
            "signal_date":       r["signal_date"],
            "direction":         r["direction"],
            "event_type":        r["event_type"] or "—",
            "strategy_bucket":   r["strategy_bucket"] or "—",
            "regime":            r["regime"] or "—",
            "gap_pct":           gap_pct,
            "rsi":               round(r.get("rsi_14") or 0, 1),
            "atr_pct":           round(atr / close * 100, 2) if close else None,
            "volume_ratio":      round(r.get("volume_ratio") or 0, 2),
            "market_conf":       r.get("market_conf"),
            "relative_opp":      r.get("relative_opp"),
            "freshness":         r.get("freshness"),
            "old_score":         round(r["old_score"], 1),
            "new_score":         round(r["new_score"], 1),
            "score_delta":       round(r["score_delta"], 1),
            "old_risk_penalty":  round(r["old_rp"], 1),
            "new_risk_penalty":  round(r["new_rp"], 1),
            "rp_delta":          round(r["rp_delta"], 1),
            "earn_strength":     round(r["earn_strength"], 1),
            "t5_pnl":            round(r["t5_pnl_pct"], 2),
            "paper_exit":        r.get("paper_exit") or "—",
            "paper_pnl":         round(r["paper_pnl_pct"], 2) if r.get("paper_pnl_pct") is not None else None,
            "reason_for_upgrade": _reason(r),
        })

    # ── Pattern summary ───────────────────────────────────────────────────────
    n = len(false_ups)
    from collections import Counter

    evt_counts     = Counter(r["event_type"] or "general" for r in false_ups)
    dir_counts     = Counter(r["direction"]  or "LONG"     for r in false_ups)
    regime_counts  = Counter(r["regime"]     or "neutral"  for r in false_ups)
    bucket_counts  = Counter(r["strategy_bucket"] or "—"   for r in false_ups)

    gap_vals    = [r.get("change_pct") or 0 for r in false_ups]
    rsi_vals    = [r.get("rsi_14")     or 0 for r in false_ups]
    atr_vals    = [round((r.get("atr_14") or 0) / (r.get("close_price") or 1) * 100, 2)
                   for r in false_ups]
    vr_vals     = [r.get("volume_ratio") or 0 for r in false_ups]

    pattern = {
        "n":                 n,
        "all_promoted_n":    sum(1 for r in rows if r["score_delta"] >= 3),
        "false_upgrade_rate": round(n / max(1, sum(1 for r in rows if r["score_delta"] >= 3)) * 100, 1),
        "top_event_types":   evt_counts.most_common(3),
        "top_directions":    dir_counts.most_common(3),
        "top_regimes":       regime_counts.most_common(3),
        "top_buckets":       bucket_counts.most_common(3),
        "avg_gap_pct":       round(sum(gap_vals) / n, 2),
        "avg_rsi":           round(sum(rsi_vals) / n, 1),
        "avg_atr_pct":       round(sum(atr_vals) / n, 2),
        "avg_volume_ratio":  round(sum(vr_vals)  / n, 2),
        "pct_earnings":      round(sum(1 for r in false_ups if r["event_type"] == "earnings") / n * 100, 1),
        "pct_short":         round(sum(1 for r in false_ups if (r["direction"] or "") == "SHORT") / n * 100, 1),
        "pct_choppy":        round(sum(1 for r in false_ups if "choppy" in (r["regime"] or "")) / n * 100, 1),
        "pct_large_gap":     round(sum(1 for r in false_ups if abs(r.get("change_pct") or 0) >= 3) / n * 100, 1),
        "pct_high_rsi_long": round(sum(1 for r in false_ups
                                       if (r.get("rsi_14") or 0) > 70
                                       and (r["direction"] or "") == "LONG") / n * 100, 1),
    }

    return {"signals": signal_rows, "pattern": pattern}


# ─────────────────────────────────────────────────────────────────────────────
# 14. Empirical Threshold Backtest
# ─────────────────────────────────────────────────────────────────────────────

def empirical_threshold_backtest(conn) -> list:
    """
    Tests multiple score thresholds and percentile cuts against historical outcomes.

    Fixed thresholds: new_score >= 50 / 52 / 55 / 58 / 60 / 62 / 65
    Percentile cuts:  top 20% / 15% / 10% / 5% of signals by new_score

    Uses new_score (rescored) so results reflect current scoring logic.
    position_size_mult is recomputed at runtime from regime + direction + price_row.

    Per threshold returns:
      label, n, win_rate, avg_t5, avg_alpha, avg_R,
      hit_target_rate, hit_stop_rate, t5_exit_rate,
      worst_t5 (max single-signal loss = max drawdown contribution),
      avg_position_size_mult
    """
    from modules.analyzer import (
        score_risk_penalty, score_post_earnings_risk,
        compute_position_size_mult,
    )
    from bisect import bisect_right
    from collections import defaultdict

    rows = conn.execute("""
        SELECT so.symbol, so.signal_date, so.t5_date, so.direction, so.regime,
               so.event_type, so.t5_pnl_pct, so.paper_pnl_pct, so.paper_exit,
               so.entry_price, so.stop_price,
               tc.final_score      AS old_score,
               tc.risk_penalty_score AS old_rp,
               ps.change_pct, ps.rsi_14, ps.atr_14, ps.close_price,
               ps.volume_ratio, ps.day_high, ps.day_low,
               ws.sector
        FROM signal_outcomes so
        JOIN trade_candidates tc
          ON tc.symbol = so.symbol AND tc.run_date = so.signal_date
        JOIN price_snapshots ps
          ON ps.symbol = so.symbol AND ps.snapshot_date = so.signal_date
        LEFT JOIN watched_symbols ws ON ws.symbol = so.symbol
        WHERE so.outcome != 'PENDING'
          AND so.t5_pnl_pct IS NOT NULL
    """).fetchall()

    if not rows:
        return []

    rows = [dict(r) for r in rows]

    # ── Rescore to new_score ──────────────────────────────────────────────────
    for r in rows:
        price_row = {k: r[k] for k in (
            "change_pct", "rsi_14", "atr_14", "close_price",
            "volume_ratio", "day_high", "day_low",
        )}
        news_scores = {
            "best_event_type": r["event_type"] or "general",
            "mixedness": 0.0,
        }
        regime_dict = {"regime": r["regime"] or "neutral"}
        direction   = r["direction"] or "LONG"
        old_rp      = r["old_rp"]    or 0.0
        old_score   = r["old_score"] or 0.0

        if (r["event_type"] or "") == "earnings":
            earn_strength, new_rp = score_post_earnings_risk(
                price_row, direction, news_scores, regime_dict
            )
        else:
            earn_strength = 0.0
            new_rp = score_risk_penalty(
                price_row, direction, news_scores, regime_dict
            )

        r["new_score"] = round(old_score + earn_strength + (old_rp - new_rp), 2)

        # Position size multiplier (recomputed from available fields)
        r["pos_mult"] = compute_position_size_mult(regime_dict, direction, price_row)

    # ── R-multiple ────────────────────────────────────────────────────────────
    def _r_mult(r):
        ep  = r.get("entry_price")
        sp  = r.get("stop_price")
        pnl = r.get("paper_pnl_pct")
        if ep and sp and ep > 0 and pnl is not None:
            risk = abs(ep - sp)
            if risk > 0:
                return round((pnl / 100.0) * ep / risk, 3)
        return None

    for r in rows:
        r["r_multiple"] = _r_mult(r)

    # ── Direction-adjusted alpha ──────────────────────────────────────────────
    price_rows = conn.execute(
        "SELECT symbol, snapshot_date, close_price FROM price_snapshots"
    ).fetchall()
    _prices = {(pr["symbol"], pr["snapshot_date"]): pr["close_price"]
               for pr in price_rows}

    _sym_dates: dict = defaultdict(list)
    for sym, d in _prices:
        _sym_dates[sym].append(d)
    for sym in _sym_dates:
        _sym_dates[sym].sort()

    def _nc(sym, target):
        exact = _prices.get((sym, target))
        if exact is not None:
            return exact
        dates = _sym_dates.get(sym)
        if not dates:
            return None
        idx = bisect_right(dates, target)
        candidates = []
        if idx > 0:          candidates.append(dates[idx - 1])
        if idx < len(dates): candidates.append(dates[idx])
        if not candidates:
            return None
        nearest = min(candidates, key=lambda d: (
            abs((date.fromisoformat(d) - date.fromisoformat(target)).days),
            d > target,
        ))
        return _prices.get((sym, nearest))

    for r in rows:
        sector    = r.get("sector") or ""
        bench_sym = (_SYMBOL_BENCHMARK.get(r["symbol"])
                     or _SECTOR_BENCHMARK.get(sector, _DEFAULT_BENCHMARK))
        sign      = -1.0 if (r.get("direction") or "LONG").upper() == "SHORT" else 1.0
        p0 = _nc(bench_sym, r["signal_date"])
        p1 = _nc(bench_sym, r["t5_date"]) if r.get("t5_date") else None
        if p0 and p1 and p0 > 0:
            bench_ret = (p1 - p0) / p0 * 100
            r["dir_adj_alpha"] = round(r["t5_pnl_pct"] - sign * bench_ret, 3)
        else:
            r["dir_adj_alpha"] = None

    # ── Win helper ────────────────────────────────────────────────────────────
    def _is_win(r):
        return (r["paper_exit"] == "HIT_TARGET"
                or (r["paper_exit"] == "T5_EXIT" and (r["paper_pnl_pct"] or 0) > 0))

    # ── Per-group stats ───────────────────────────────────────────────────────
    def _stats(subset, label):
        n = len(subset)
        if n == 0:
            return {"label": label, "n": 0}

        wins        = sum(1 for r in subset if _is_win(r))
        hit_target  = sum(1 for r in subset if r["paper_exit"] == "HIT_TARGET")
        hit_stop    = sum(1 for r in subset if r["paper_exit"] == "HIT_STOP")
        t5_exit     = sum(1 for r in subset if r["paper_exit"] == "T5_EXIT")

        t5_vals     = [r["t5_pnl_pct"]    for r in subset if r["t5_pnl_pct"]    is not None]
        alpha_vals  = [r["dir_adj_alpha"] for r in subset if r["dir_adj_alpha"] is not None]
        r_vals      = [r["r_multiple"]    for r in subset if r["r_multiple"]    is not None]
        mult_vals   = [r["pos_mult"]      for r in subset]

        return {
            "label":              label,
            "n":                  n,
            "win_rate":           round(wins / n * 100, 1),
            "avg_t5":             _safe_avg(t5_vals),
            "avg_alpha":          _safe_avg(alpha_vals),
            "avg_R":              _safe_avg(r_vals),
            "hit_target_rate":    round(hit_target / n * 100, 1),
            "hit_stop_rate":      round(hit_stop   / n * 100, 1),
            "t5_exit_rate":       round(t5_exit    / n * 100, 1),
            "worst_t5":           round(min(t5_vals), 2) if t5_vals else None,
            "avg_pos_mult":       _safe_avg(mult_vals),
        }

    results = []

    # Baseline: all signals
    results.append(_stats(rows, "ALL (baseline)"))

    # Fixed score thresholds
    for thresh in (50, 52, 55, 58, 60, 62, 65):
        subset = [r for r in rows if r["new_score"] >= thresh]
        results.append(_stats(subset, f"score ≥ {thresh}"))

    # Percentile cuts
    all_scores = sorted(r["new_score"] for r in rows)
    total = len(all_scores)
    for pct in (20, 15, 10, 5):
        cutoff_idx = max(0, total - round(total * pct / 100))
        if cutoff_idx < total:
            min_score = all_scores[cutoff_idx]
            subset = [r for r in rows if r["new_score"] >= min_score]
            results.append(_stats(subset, f"top {pct}% (≥{min_score:.1f})"))
        else:
            results.append({"label": f"top {pct}%", "n": 0})

    return results
