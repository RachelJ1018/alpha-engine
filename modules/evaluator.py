"""
evaluator.py — evaluation reports for Alpha Engine

1. signal_stability_report(conn, days)  → signal volume, score distribution, event mix
2. score_return_buckets(conn)           → does higher score → better return?
3. paper_trade_summary(conn)            → exit distribution, win rate, P&L stats

All functions accept a sqlite3 connection (row_factory=sqlite3.Row) and return
plain dicts/lists safe to render directly in Streamlit.
"""

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
    ("<50",   0,   50),
    ("50-59", 50,  60),
    ("60-69", 60,  70),
    ("70-79", 70,  80),
    ("≥80",   80, 200),
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
