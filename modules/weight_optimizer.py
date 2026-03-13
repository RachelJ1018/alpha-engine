"""
weight_optimizer.py — Analyzes signal_outcomes to find what actually predicts
win rate, then emits recommended weight adjustments for analyzer.py.

Usage:
    python -m modules.weight_optimizer           # print analysis + recommendations
    python -m modules.weight_optimizer --apply   # also write weights to weights.json
    python -m modules.weight_optimizer --min 10  # require ≥10 resolved samples

What it does:
  1. Breaks down win rate by event_type (from trade_candidates joined to outcomes)
  2. Breaks down win rate by score band (already in evaluator, but here we add
     statistical significance flags and per-component correlation)
  3. Computes per-layer correlation with t5_pnl to rank which scoring layers
     are actually predictive
  4. Emits EVENT_IMPORTANCE overrides and layer multipliers as a JSON file
     that analyzer.py can optionally load on startup.

Interpretation guide (printed with the report):
  • If earnings win rate >> general/macro → increase earnings EVENT_IMPORTANCE
  • If event_edge correlation is weak but market_conf is strong → consider
    rebalancing max points (e.g. event_edge 20pts, market_conf 25pts)
  • Layer multipliers scale the raw layer score before it enters compute_final_score
    (multiplier 1.0 = no change; 1.2 = boost; 0.8 = dampen)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from typing import Any

from modules.db import get_conn, DB_PATH

# ── Constants ────────────────────────────────────────────────────────────────

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "weights.json")

# Minimum resolved samples in a bucket to trust its stats
MIN_SAMPLE_DEFAULT = 5

# Layer fields stored on trade_candidates (joined with signal_outcomes)
LAYER_FIELDS = [
    "event_edge_score",
    "market_conf_score",
    "regime_fit_score",
    "relative_opp_score",
    "freshness_score",
    "risk_penalty_score",   # higher = worse; correlation should be negative
]

EVENT_TYPES = ["earnings", "macro", "ma", "ai", "regulation", "layoff", "product", "general"]

# ── Helpers ──────────────────────────────────────────────────────────────────

def _avg(vals: list) -> float | None:
    v = [x for x in vals if x is not None]
    return round(sum(v) / len(v), 3) if v else None

def _win_rate(rows: list) -> float | None:
    resolved = [r for r in rows if r.get("outcome") not in ("PENDING", None)]
    if not resolved:
        return None
    wins = sum(1 for r in resolved if r.get("outcome") == "WIN")
    return round(wins / len(resolved) * 100, 1)

def _pearson(xs: list, ys: list) -> float | None:
    """Simple Pearson r between two lists (same length, no Nones)."""
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx  = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy  = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return None
    return round(num / (dx * dy), 3)

def _significance_flag(n: int, rate: float | None) -> str:
    """Return a rough significance indicator."""
    if n < 5:
        return "⚠ too few samples"
    if n < 15:
        return "~ low confidence"
    return "✓ reliable" if rate is not None else ""

# ── Core analysis ─────────────────────────────────────────────────────────────

def _load_joined_data(conn) -> list[dict]:
    """
    Join signal_outcomes with trade_candidates to get per-signal layer scores
    and the event_type that drove it.
    """
    rows = conn.execute("""
        SELECT
            so.symbol,
            so.signal_date,
            so.outcome,
            so.t5_pnl_pct,
            so.paper_pnl_pct,
            so.paper_exit,
            so.regime,
            so.strategy_bucket,
            so.final_score,
            tc.event_edge_score,
            tc.market_conf_score,
            tc.regime_fit_score,
            tc.relative_opp_score,
            tc.freshness_score,
            tc.risk_penalty_score
        FROM signal_outcomes so
        LEFT JOIN trade_candidates tc
            ON  so.symbol      = tc.symbol
            AND so.signal_date = tc.run_date
        WHERE so.outcome != 'PENDING'
        ORDER BY so.signal_date ASC
    """).fetchall()
    return [dict(r) for r in rows]


def _load_event_data(conn) -> list[dict]:
    """
    Pull event_type directly from signal_outcomes (stored at signal time).
    Falls back to strategy_bucket mapping for rows recorded before event_type column was added.
    """
    rows = conn.execute("""
        SELECT
            so.outcome,
            so.t5_pnl_pct,
            so.paper_pnl_pct,
            so.paper_exit,
            so.event_type,
            so.strategy_bucket,
            so.regime,
            so.final_score
        FROM signal_outcomes so
        WHERE so.outcome != 'PENDING'
    """).fetchall()
    return [dict(r) for r in rows]


# ── Section 1: event-type win rates ──────────────────────────────────────────

def analyze_event_types(rows: list[dict], min_n: int) -> dict[str, Any]:
    """
    Group by event_type (from signal_outcomes.event_type, stored at signal time).
    Falls back to strategy_bucket mapping for legacy rows without event_type.
    Returns results dict and recommended EVENT_IMPORTANCE overrides.
    """
    # Fallback mapping for rows that predate the event_type column.
    # strategy_bucket → event_type is intentionally conservative:
    # event_long/event_short are catch-alls, so they map to "general" not "macro".
    BUCKET_TO_EVENT = {
        "post_earnings_drift": "earnings",
        "event_long":          "general",
        "event_short":         "general",
        "sympathy_play":       "product",
        "macro_watch":         "macro",
        "opinion_watch":       "general",
    }

    groups: dict[str, list] = {}
    for r in rows:
        evt = (r.get("event_type")
               or BUCKET_TO_EVENT.get(r.get("strategy_bucket") or "", "general"))
        groups.setdefault(evt, []).append(r)

    results = {}
    for evt, erows in sorted(groups.items(), key=lambda x: -len(x[1])):
        resolved = [r for r in erows if r.get("outcome") not in ("PENDING", None)]
        n = len(resolved)
        wr = _win_rate(resolved)
        results[evt] = {
            "n":             n,
            "win_rate":      wr,
            "avg_t5_pnl":    _avg([r["t5_pnl_pct"] for r in resolved]),
            "avg_paper_pnl": _avg([r["paper_pnl_pct"] for r in resolved]),
            "confidence":    _significance_flag(n, wr),
        }

    # Current defaults (mirrored from analyzer.py)
    current = {
        "earnings": 1.00, "macro": 0.85, "ma": 0.85, "ai": 0.80,
        "regulation": 0.75, "layoff": 0.60, "product": 0.55, "general": 0.20,
    }

    # Build EVENT_IMPORTANCE overrides via 60/40 blend: 60% prior + 40% data signal
    overrides: dict[str, float] = {}
    for evt, stats in results.items():
        if stats["n"] < min_n or stats["win_rate"] is None:
            continue
        wr = stats["win_rate"] / 100.0
        # Map win_rate 0→1 linearly to importance 0.10→1.00
        data_signal = 0.10 + wr * 0.90
        blended = round(0.6 * current.get(evt, 0.50) + 0.4 * data_signal, 2)
        overrides[evt] = blended

    return {"by_event": results, "event_importance_overrides": overrides}


# ── Section 2: score-band win rates ──────────────────────────────────────────

SCORE_BANDS = [
    ("<55",   0,   55),
    ("55-64", 55,  65),
    ("65-74", 65,  75),
    ("75-84", 75,  85),
    ("≥85",   85, 200),
]

def analyze_score_bands(rows: list[dict], min_n: int) -> list[dict]:
    results = []
    for label, lo, hi in SCORE_BANDS:
        subset = [r for r in rows if r.get("final_score") is not None
                  and lo <= r["final_score"] < hi]
        resolved = [r for r in subset if r.get("outcome") not in ("PENDING", None)]
        n = len(resolved)
        wr = _win_rate(resolved)
        results.append({
            "band":        label,
            "n":           n,
            "win_rate":    wr,
            "avg_t5_pnl":  _avg([r["t5_pnl_pct"] for r in resolved]),
            "avg_paper_pnl": _avg([r.get("paper_pnl_pct") for r in resolved]),
            "confidence":  _significance_flag(n, wr),
        })
    return results


# ── Section 3: layer correlations ────────────────────────────────────────────

def analyze_layer_correlations(rows: list[dict], min_n: int) -> dict[str, Any]:
    """
    Pearson correlation of each scoring layer with t5_pnl_pct.
    Negative for risk_penalty is expected (more penalty → worse outcome).
    Returns correlations + recommended layer multipliers.
    """
    usable = [r for r in rows if r.get("t5_pnl_pct") is not None]
    if len(usable) < min_n:
        return {"note": f"Only {len(usable)} resolved samples — need {min_n} for correlations."}

    t5_pnls = [r["t5_pnl_pct"] for r in usable]
    correlations: dict[str, Any] = {}

    for field in LAYER_FIELDS:
        layer_vals = [r.get(field) for r in usable]
        paired = [(x, y) for x, y in zip(layer_vals, t5_pnls) if x is not None]
        if len(paired) < min_n:
            correlations[field] = {"r": None, "n": len(paired), "note": "too few non-null values"}
            continue
        xs, ys = zip(*paired)
        r = _pearson(list(xs), list(ys))
        correlations[field] = {
            "r":    r,
            "n":    len(paired),
            "note": ("negative as expected" if field == "risk_penalty_score" and r and r < 0
                     else "unexpectedly positive" if field == "risk_penalty_score" and r and r > 0
                     else ""),
        }

    # Recommend layer multipliers
    # Logic: amplify layers with |r| > 0.15 in the right direction,
    #        dampen layers with |r| < 0.05 (noise) or wrong direction.
    multipliers: dict[str, float] = {}
    for field, stats in correlations.items():
        r = stats.get("r")
        if r is None:
            multipliers[field] = 1.0
            continue
        # risk_penalty is subtracted, so negative r is "working correctly"
        effective_r = -r if field == "risk_penalty_score" else r
        if effective_r > 0.20:
            multipliers[field] = round(min(effective_r * 5 + 1.0, 1.5), 2)  # cap at 1.5x
        elif effective_r > 0.10:
            multipliers[field] = 1.1
        elif effective_r < -0.10:
            multipliers[field] = round(max(effective_r * 3 + 1.0, 0.5), 2)  # floor at 0.5x
        else:
            multipliers[field] = 1.0  # neutral — no change

    return {"correlations": correlations, "layer_multipliers": multipliers}


# ── Section 4: regime breakdown ───────────────────────────────────────────────

def analyze_regime(rows: list[dict], min_n: int) -> dict[str, Any]:
    regimes: dict[str, list] = {}
    for r in rows:
        reg = r.get("regime") or "unknown"
        regimes.setdefault(reg, []).append(r)

    results = {}
    for reg, rrows in regimes.items():
        resolved = [r for r in rrows if r.get("outcome") not in ("PENDING", None)]
        n = len(resolved)
        results[reg] = {
            "n":           n,
            "win_rate":    _win_rate(resolved),
            "avg_t5_pnl":  _avg([r["t5_pnl_pct"] for r in resolved]),
            "confidence":  _significance_flag(n, _win_rate(resolved)),
        }
    return results


# ── Output builder ────────────────────────────────────────────────────────────

def build_weights_json(event_overrides: dict, layer_multipliers: dict) -> dict:
    return {
        "_meta": {
            "generated_by": "weight_optimizer.py",
            "note": "Loaded by analyzer.py on startup to override static defaults. "
                    "Delete this file to revert to hardcoded defaults.",
        },
        "event_importance": event_overrides,
        "layer_multipliers": layer_multipliers,
    }


def save_weights(weights: dict) -> str:
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    with open(WEIGHTS_PATH, "w") as f:
        json.dump(weights, f, indent=2)
    return WEIGHTS_PATH


# ── Pretty printer ────────────────────────────────────────────────────────────

def _bar(value: float | None, max_val: float = 100, width: int = 20) -> str:
    if value is None:
        return " " * width
    filled = int((value / max_val) * width)
    return "█" * filled + "░" * (width - filled)

def print_report(
    event_analysis: dict,
    score_bands: list,
    layer_analysis: dict,
    regime_analysis: dict,
    weights: dict,
):
    SEP = "─" * 70

    print(f"\n{'═' * 70}")
    print("  ALPHA ENGINE — WEIGHT OPTIMIZER REPORT")
    print(f"{'═' * 70}\n")

    # ── Event type win rates ──
    print("1. WIN RATE BY EVENT TYPE")
    print(SEP)
    print(f"  {'Event type':<16}  {'N':>4}  {'Win%':>6}  {'Avg t+5':>8}  {'Confidence'}")
    print(f"  {'─'*16}  {'─'*4}  {'─'*6}  {'─'*8}  {'─'*18}")
    for evt, stats in event_analysis["by_event"].items():
        wr  = f"{stats['win_rate']:.1f}%" if stats['win_rate'] is not None else "  —  "
        pnl = f"{stats['avg_t5_pnl']:+.2f}%" if stats['avg_t5_pnl'] is not None else "   —   "
        print(f"  {evt:<16}  {stats['n']:>4}  {wr:>6}  {pnl:>8}  {stats['confidence']}")

    if event_analysis["event_importance_overrides"]:
        print(f"\n  Recommended EVENT_IMPORTANCE overrides:")
        current = {
            "earnings": 1.00, "macro": 0.85, "ma": 0.85, "ai": 0.80,
            "regulation": 0.75, "layoff": 0.60, "product": 0.55, "general": 0.20,
        }
        for evt, val in sorted(event_analysis["event_importance_overrides"].items()):
            arrow = "↑" if val > current.get(evt, 0.5) else "↓" if val < current.get(evt, 0.5) else "="
            print(f"    {evt:<12}: {current.get(evt, '?'):.2f} → {val:.2f}  {arrow}")
    else:
        print("\n  (not enough data to recommend overrides yet)")

    # ── Score bands ──
    print(f"\n\n2. WIN RATE BY SCORE BAND")
    print(SEP)
    print(f"  {'Band':<8}  {'N':>4}  {'Win%':>6}  {'Avg t+5':>8}  {'Bar':<22}  {'Confidence'}")
    print(f"  {'─'*8}  {'─'*4}  {'─'*6}  {'─'*8}  {'─'*22}  {'─'*18}")
    for band in score_bands:
        wr  = band["win_rate"]
        pnl = band["avg_t5_pnl"]
        bar = _bar(wr, 100, 20) if wr is not None else " " * 20
        wr_str  = f"{wr:.1f}%" if wr is not None else "  —  "
        pnl_str = f"{pnl:+.2f}%" if pnl is not None else "   —   "
        print(f"  {band['band']:<8}  {band['n']:>4}  {wr_str:>6}  {pnl_str:>8}  {bar}  {band['confidence']}")

    # ── Layer correlations ──
    print(f"\n\n3. LAYER CORRELATION WITH t+5 P&L")
    print(SEP)
    if "note" in layer_analysis:
        print(f"  {layer_analysis['note']}")
    else:
        corr = layer_analysis["correlations"]
        mults = layer_analysis["layer_multipliers"]
        print(f"  {'Layer':<24}  {'r':>6}  {'N':>4}  {'Multiplier':>10}  Notes")
        print(f"  {'─'*24}  {'─'*6}  {'─'*4}  {'─'*10}  {'─'*24}")
        for field in LAYER_FIELDS:
            s = corr.get(field, {})
            r_val = s.get("r")
            r_str = f"{r_val:+.3f}" if r_val is not None else "   — "
            mult  = mults.get(field, 1.0)
            m_str = f"{mult:.2f}×"
            m_indicator = "↑ boost" if mult > 1.05 else "↓ dampen" if mult < 0.95 else "= keep"
            note = s.get("note", "")
            print(f"  {field:<24}  {r_str:>6}  {s.get('n',0):>4}  {m_str:>10}  {m_indicator}  {note}")

    # ── Regime ──
    print(f"\n\n4. WIN RATE BY MARKET REGIME")
    print(SEP)
    print(f"  {'Regime':<10}  {'N':>4}  {'Win%':>6}  {'Avg t+5':>8}  {'Confidence'}")
    print(f"  {'─'*10}  {'─'*4}  {'─'*6}  {'─'*8}  {'─'*18}")
    for reg, stats in regime_analysis.items():
        wr  = f"{stats['win_rate']:.1f}%" if stats['win_rate'] is not None else "  —  "
        pnl = f"{stats['avg_t5_pnl']:+.2f}%" if stats['avg_t5_pnl'] is not None else "   —   "
        print(f"  {reg:<10}  {stats['n']:>4}  {wr:>6}  {pnl:>8}  {stats['confidence']}")

    # ── Recommended weights ──
    print(f"\n\n5. RECOMMENDED WEIGHTS  (write to data/weights.json)")
    print(SEP)
    print("  EVENT_IMPORTANCE:")
    for k, v in weights["event_importance"].items():
        print(f"    {k:<12}: {v}")
    print("\n  LAYER MULTIPLIERS:")
    for k, v in weights["layer_multipliers"].items():
        print(f"    {k:<28}: {v}×")

    print(f"\n{'═' * 70}")
    print("  HOW TO APPLY")
    print(f"{'═' * 70}")
    print("""
  Option A (automatic):
      python -m modules.weight_optimizer --apply
      → writes data/weights.json; analyzer.py loads it on next run.

  Option B (manual):
      Edit EVENT_IMPORTANCE in modules/analyzer.py directly using the
      values printed above — gives you full control over each change.

  Either way, run your backtest after to validate the update:
      python -m modules.backtest --tickers NVDA TSLA --years 2
""")


# ── Main ─────────────────────────────────────────────────────────────────────

def run(min_n: int = MIN_SAMPLE_DEFAULT, apply: bool = False):
    conn = get_conn()

    # Check if we have any resolved data at all
    total = conn.execute(
        "SELECT COUNT(*) FROM signal_outcomes WHERE outcome != 'PENDING'"
    ).fetchone()[0]

    if total == 0:
        print("\n⚠  No resolved signal outcomes yet.")
        print("   The system needs at least a few weeks of daily runs before")
        print("   signal_outcomes has enough resolved (non-PENDING) rows.")
        print("   Come back once you have some WIN/LOSS/SCRATCH outcomes!\n")
        conn.close()
        return

    print(f"\n[weight_optimizer] {total} resolved outcomes found in signal_outcomes.")

    joined  = _load_joined_data(conn)
    events  = _load_event_data(conn)
    conn.close()

    event_analysis = analyze_event_types(events, min_n)
    score_bands    = analyze_score_bands(joined, min_n)
    layer_analysis = analyze_layer_correlations(joined, min_n)
    regime_analysis = analyze_regime(joined, min_n)

    # Build weights JSON
    layer_multipliers = layer_analysis.get("layer_multipliers", {f: 1.0 for f in LAYER_FIELDS})
    event_overrides   = event_analysis.get("event_importance_overrides", {})
    weights = build_weights_json(event_overrides, layer_multipliers)

    print_report(event_analysis, score_bands, layer_analysis, regime_analysis, weights)

    if apply:
        path = save_weights(weights)
        print(f"✓ Weights written to {path}")
        print("  analyzer.py will load these on next run.\n")
    else:
        print("  Run with --apply to save these weights automatically.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alpha Engine weight optimizer")
    parser.add_argument("--apply",  action="store_true",
                        help="Write recommended weights to data/weights.json")
    parser.add_argument("--min",    type=int, default=MIN_SAMPLE_DEFAULT,
                        help=f"Minimum sample size per bucket (default: {MIN_SAMPLE_DEFAULT})")
    args = parser.parse_args()
    run(min_n=args.min, apply=args.apply)