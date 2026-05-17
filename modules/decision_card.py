"""
decision_card.py — Layer 2: Decision Card logic.

Translates Layer 1 scored signals into structured, human-readable decision data.
No Streamlit here — pure computation. Rendering lives in app.py.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple


# ── Volume + MarketConf bucketing ──────────────────────────────────────────────

def bucket_volume(vr: Optional[float]) -> str:
    if vr is None:
        return "unknown"
    vr = float(vr)
    if vr < 0.5:  return "vr_<0.5"
    if vr < 1.0:  return "vr_0.5_1.0"
    if vr < 1.5:  return "vr_1.0_1.5"
    return "vr_>=1.5"


def bucket_market_conf(mc: Optional[float]) -> str:
    if mc is None:
        return "unknown"
    mc = float(mc)
    if mc < 5:   return "mc_<5"
    if mc < 10:  return "mc_5_10"
    if mc < 15:  return "mc_10_15"
    return "mc_>=15"


# ── Evidence level ─────────────────────────────────────────────────────────────

def evidence_level(n: int, avg_t5: float, hit_stop_rate: float) -> str:
    """INSUFFICIENT / EARLY_POSITIVE / EARLY_WEAK / DEVELOPING_POSITIVE /
    DEVELOPING_WEAK / RELIABLE_POSITIVE / RELIABLE_WEAK"""
    if n < 5:
        return "INSUFFICIENT"
    positive = avg_t5 > 0 and hit_stop_rate < 40
    if n < 15:
        return "EARLY_POSITIVE" if positive else "EARLY_WEAK"
    if n < 30:
        return "DEVELOPING_POSITIVE" if positive else "DEVELOPING_WEAK"
    return "RELIABLE_POSITIVE" if positive else "RELIABLE_WEAK"


def evidence_interpretation(ev_level: str, n: int) -> str:
    return {
        "INSUFFICIENT":        f"Only {n} resolved signal(s) — too few to draw conclusions.",
        "EARLY_POSITIVE":      f"n={n}, early signs are positive. Promising, not proven.",
        "EARLY_WEAK":          f"n={n}, early results are mixed or negative. Proceed with caution.",
        "DEVELOPING_POSITIVE": f"n={n}, developing positive track record. Gaining confidence.",
        "DEVELOPING_WEAK":     f"n={n}, results are inconsistent. Re-examine setup criteria.",
        "RELIABLE_POSITIVE":   f"n={n}, statistically meaningful positive alpha. Use this edge.",
        "RELIABLE_WEAK":       f"n={n}, sufficient history but edge is weak or negative. Avoid.",
    }.get(ev_level, f"n={n}")


# ── Historical evidence ────────────────────────────────────────────────────────

def _compute_stats(rows: list) -> Optional[Dict[str, Any]]:
    t5s = [float(r["t5_pnl_pct"]) for r in rows if r["t5_pnl_pct"] is not None]
    if not t5s:
        return None
    n = len(t5s)
    wins = sum(1 for t in t5s if t > 0)
    hit_stop   = sum(1 for r in rows if (r["paper_exit"] or "") == "HIT_STOP")
    hit_target = sum(1 for r in rows if (r["paper_exit"] or "") == "HIT_TARGET")
    t5_exit    = sum(1 for r in rows if (r["paper_exit"] or "") == "T5_EXIT")
    return {
        "n":               n,
        "win_rate":        round(wins / n * 100, 1),
        "avg_t5":          round(sum(t5s) / n, 2),
        "worst_t5":        round(min(t5s), 2),
        "hit_stop_rate":   round(hit_stop / n * 100, 1),
        "hit_target_rate": round(hit_target / n * 100, 1),
        "t5_exit_rate":    round(t5_exit / n * 100, 1),
    }


def get_historical_evidence(
    conn,
    strategy_bucket: str,
    direction: str,
    volume_ratio: Optional[float],
    market_conf: Optional[float],
    regime: str,
) -> Dict[str, Any]:
    """Query resolved signal_outcomes for similar setups. Progressive relaxation.

    Match levels (most → least specific):
      1. bucket + direction + regime + vol_bucket + mc_bucket
      2. bucket + direction + regime
      3. bucket + direction (all regimes)
    """
    _EMPTY = {
        "n": 0, "win_rate": 0, "avg_t5": 0, "worst_t5": 0,
        "hit_stop_rate": 0, "hit_target_rate": 0, "t5_exit_rate": 0,
        "match_level": "none", "match_desc": "no similar resolved signals yet",
        "evidence_level": "INSUFFICIENT",
        "interpretation": "No resolved signals for this setup yet.",
    }

    if not strategy_bucket:
        return _EMPTY

    rows = conn.execute("""
        SELECT so.t5_pnl_pct, so.paper_exit, so.regime,
               ps.volume_ratio AS vr, tc.market_conf_score AS mc
        FROM signal_outcomes so
        LEFT JOIN price_snapshots ps
          ON so.symbol = ps.symbol AND so.signal_date = ps.snapshot_date
        LEFT JOIN trade_candidates tc
          ON so.symbol = tc.symbol AND so.signal_date = tc.run_date
        WHERE so.t5_pnl_pct IS NOT NULL
          AND so.strategy_bucket = ?
          AND so.direction = ?
        ORDER BY so.signal_date DESC
        LIMIT 100
    """, (strategy_bucket, direction)).fetchall()

    if not rows:
        return _EMPTY

    vol_bkt = bucket_volume(volume_ratio)
    mc_bkt  = bucket_market_conf(market_conf)

    candidates = [
        (
            lambda r, _r=regime, _v=vol_bkt, _m=mc_bkt: (
                r["regime"] == _r
                and bucket_volume(r["vr"])      == _v
                and bucket_market_conf(r["mc"]) == _m
            ),
            f"{strategy_bucket} · {direction} · {regime} · vol={vol_bkt} · MC={mc_bkt}",
            "exact",
        ),
        (
            lambda r, _r=regime: r["regime"] == _r,
            f"{strategy_bucket} · {direction} · regime={regime}",
            "regime",
        ),
        (
            lambda r: True,
            f"{strategy_bucket} · {direction} (all regimes)",
            "bucket",
        ),
    ]

    for filter_fn, desc, level in candidates:
        filtered = [r for r in rows if filter_fn(r)]
        if len(filtered) >= 3:
            stats = _compute_stats(filtered)
            if stats:
                ev = evidence_level(stats["n"], stats["avg_t5"], stats["hit_stop_rate"])
                return {
                    **stats,
                    "match_level": level,
                    "match_desc":  desc,
                    "evidence_level": ev,
                    "interpretation": evidence_interpretation(ev, stats["n"]),
                }

    return _EMPTY


# ── Confirmation traffic lights ────────────────────────────────────────────────

_ICON = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}

def confirmation_lights(
    catalyst_quality: str,
    volume_ratio: Optional[float],
    market_conf: Optional[float],
    regime: str,
    direction: str,
    risk_penalty: Optional[float],
) -> Dict[str, Tuple[str, str, str]]:
    """Returns dict of {label: (color, icon, description)}."""
    lights: Dict[str, Tuple[str, str, str]] = {}
    vr = float(volume_ratio or 0)
    mc = float(market_conf  or 0)
    rp = float(risk_penalty or 0)

    # ── Catalyst ──────────────────────────────────────────────────────────────
    _cq_map = {
        "STRONG": ("GREEN",  "Hard ticker-specific event from reliable source"),
        "MEDIUM": ("YELLOW", "Real catalyst but financial impact unquantified"),
        "WEAK":   ("RED",    "Macro narrative or sympathy — no company-specific edge"),
        "NONE":   ("RED",    "No catalyst — pure technical setup"),
    }
    c_color, c_desc = _cq_map.get(catalyst_quality, ("RED", catalyst_quality or "Unknown"))
    lights["Catalyst"] = (c_color, _ICON[c_color], c_desc)

    # ── Volume ────────────────────────────────────────────────────────────────
    if vr >= 1.3:
        lights["Volume"] = ("GREEN",  _ICON["GREEN"],  f"{vr:.2f}x avg — institutional participation")
    elif vr >= 0.8:
        lights["Volume"] = ("YELLOW", _ICON["YELLOW"], f"{vr:.2f}x avg — partial confirmation")
    else:
        lights["Volume"] = ("RED",    _ICON["RED"],    f"{vr:.2f}x avg — weak, do not chase")

    # ── MarketConf ────────────────────────────────────────────────────────────
    if mc >= 12:
        lights["MarketConf"] = ("GREEN",  _ICON["GREEN"],  f"{mc:.1f}/20 — strong price + technical alignment")
    elif mc >= 8:
        lights["MarketConf"] = ("YELLOW", _ICON["YELLOW"], f"{mc:.1f}/20 — moderate alignment")
    else:
        lights["MarketConf"] = ("RED",    _ICON["RED"],    f"{mc:.1f}/20 — weak price confirmation")

    # ── Regime ────────────────────────────────────────────────────────────────
    favored = (regime == "bull" and direction == "LONG") or (regime == "bear" and direction == "SHORT")
    counter = (regime == "bear" and direction == "LONG") or (regime == "bull" and direction == "SHORT")
    if favored:
        lights["Regime"] = ("GREEN",  _ICON["GREEN"],  f"{regime.upper()} regime favors {direction}")
    elif counter:
        lights["Regime"] = ("RED",    _ICON["RED"],    f"{regime.upper()} regime is counter to {direction}")
    else:
        lights["Regime"] = ("YELLOW", _ICON["YELLOW"], f"{regime.upper()} — catalyst may override macro")

    # ── Risk penalty ──────────────────────────────────────────────────────────
    if rp <= 3:
        lights["Risk"] = ("GREEN",  _ICON["GREEN"],  f"Low risk penalty ({rp:.1f}/15)")
    elif rp <= 7:
        lights["Risk"] = ("YELLOW", _ICON["YELLOW"], f"Moderate risk penalty ({rp:.1f}/15)")
    else:
        lights["Risk"] = ("RED",    _ICON["RED"],    f"High risk penalty ({rp:.1f}/15) — reduce size or skip")

    return lights


# ── Invalidation rules ─────────────────────────────────────────────────────────

def generate_invalidation(
    direction: str,
    strategy_bucket: str,
    sym: str,
    stop_price: Optional[float],
) -> List[str]:
    stop_str = f"${stop_price:.2f}" if stop_price else "the stop level"
    rules: List[str] = []

    if direction == "LONG":
        rules.append(f"Price closes below {stop_str}")
        rules.append(f"{sym} underperforms QQQ/SPY for 2 consecutive days without a new catalyst")
        rules.append("New company-specific negative news contradicts the thesis")
    else:
        rules.append(f"Price closes above {stop_str}")
        rules.append(f"{sym} outperforms benchmark for 2 consecutive days")
        rules.append("New positive news contradicts the short thesis")

    if strategy_bucket == "post_earnings_drift":
        rules.append("Earnings gap reverses intraday within 1–2 days on elevated selling volume")
    elif strategy_bucket in ("event_long", "event_short"):
        rules.append("Catalyst is retracted, denied, or contradicted by follow-up news")
    elif strategy_bucket == "mean_reversion_long":
        rules.append("No bounce by T+2 — mean-reversion failed to materialize")
    elif strategy_bucket == "relative_strength_long":
        rules.append("Stock begins to underperform its sector ETF on consecutive closes")

    rules.append("Market opens sharply risk-off (SPY pre-market ≤ −1.5%)")
    rules.append("Price gaps too far from planned entry at open (>1% away from plan)")

    return rules


# ── Verdict ────────────────────────────────────────────────────────────────────

def verdict(action: str, catalyst_quality: str, ev_level: str) -> str:
    if action == "ACTIONABLE" and catalyst_quality == "STRONG":
        return "Strong event-driven setup. Worth manual review before entry."
    if action == "ACTIONABLE" and catalyst_quality == "MEDIUM":
        return "Moderate catalyst with confirmed technicals. Verify catalyst before acting."
    if action == "ACTIONABLE":
        return "Score passes threshold but catalyst is weak. Research manually before trading."
    if action == "WATCHLIST":
        return "On watch. Do not trade until additional confirmation appears."
    return "Monitoring only. Insufficient quality to act right now."
