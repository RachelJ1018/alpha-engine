"""
signal_tracker.py — records every ACTIONABLE/WATCHLIST/MONITOR signal and
tracks price outcomes at t+1, t+3, t+5 trading days.

Paper trade model (ATR-based, independent of sidebar settings):
  stop     = entry ± 1.2 × ATR
  target   = entry ± 2.0 × stop_distance   (2R)
  exit     = first day close crosses stop or target; otherwise t+5 close

Outcome labels (based on paper_exit):
  HIT_TARGET  → closed at target (WIN)
  HIT_STOP    → closed at stop   (LOSS)
  T5_EXIT     → neither; mark at t+5 close (WIN if pnl>0, else LOSS)
  PENDING     → waiting for price data
"""

from datetime import date
from modules.db import get_conn

TRACK_ACTIONS  = {"ACTIONABLE", "WATCHLIST", "MONITOR"}
ATR_MULTIPLIER = 1.2
REWARD_RATIO   = 2.0   # 2R target

_EVENT_PRIORITY = {
    "earnings": 10, "ma": 9, "regulation": 8, "ai": 7,
    "product": 6, "macro": 5, "layoff": 4, "general": 1,
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _best_event_type(conn, symbol: str) -> str:
    """Return the highest-priority event_type from recent news for symbol."""
    rows = conn.execute("""
        SELECT event_type FROM news_articles
        WHERE symbols LIKE ?
          AND published_at >= datetime('now', '-48 hours')
          AND event_type IS NOT NULL
        ORDER BY importance_score DESC
        LIMIT 10
    """, (f'%"{symbol}"%',)).fetchall()
    types = [r["event_type"] for r in rows if r["event_type"]]
    if not types:
        return "general"
    return max(types, key=lambda e: _EVENT_PRIORITY.get(e.lower(), 0))

def _pnl_pct(entry: float, price: float, direction: str) -> float:
    if not entry or not price:
        return None
    raw = (price - entry) / entry * 100.0
    return round(raw if direction == "LONG" else -raw, 3)


def _price_at_n(conn, symbol: str, signal_date: str, n: int):
    """(price, date) at the nth trading day after signal_date, or (None, None)."""
    rows = conn.execute("""
        SELECT snapshot_date, close_price FROM price_snapshots
        WHERE symbol = ? AND snapshot_date > ?
          AND close_price IS NOT NULL
        ORDER BY snapshot_date ASC
    """, (symbol, signal_date)).fetchall()
    if len(rows) >= n:
        r = rows[n - 1]
        return float(r["close_price"]), r["snapshot_date"]
    return None, None


def _all_prices_after(conn, symbol: str, signal_date: str, max_days: int = 10):
    """Return list of (close_price, date, day_high, day_low) for up to max_days trading days."""
    rows = conn.execute("""
        SELECT snapshot_date, close_price, day_high, day_low FROM price_snapshots
        WHERE symbol = ? AND snapshot_date > ?
          AND close_price IS NOT NULL
        ORDER BY snapshot_date ASC LIMIT ?
    """, (symbol, signal_date, max_days)).fetchall()
    return [
        (
            float(r["close_price"]),
            r["snapshot_date"],
            float(r["day_high"]) if r["day_high"] else None,
            float(r["day_low"])  if r["day_low"]  else None,
        )
        for r in rows
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Record signals
# ─────────────────────────────────────────────────────────────────────────────

def record_signals(regime: dict, run_date: str = None, verbose: bool = True) -> int:
    conn = get_conn()
    if run_date is None:
        run_date = date.today().isoformat()

    regime_label = regime.get("regime", "unknown")

    candidates = conn.execute("""
        SELECT tc.symbol, tc.action, tc.final_score, tc.direction,
               tc.strategy_bucket, ps.close_price, ps.atr_14
        FROM trade_candidates tc
        LEFT JOIN price_snapshots ps
          ON tc.symbol = ps.symbol AND ps.snapshot_date = ?
        WHERE tc.run_date = ? AND tc.action IN ('ACTIONABLE','WATCHLIST','MONITOR')
    """, (run_date, run_date)).fetchall()

    saved = 0
    for c in candidates:
        entry = c["close_price"]
        if not entry:
            continue

        atr  = c["atr_14"] or 0.0
        stop_dist   = max(atr * ATR_MULTIPLIER, entry * 0.02)  # floor at 2%
        target_dist = stop_dist * REWARD_RATIO

        if c["direction"] == "LONG":
            stop_price   = round(entry - stop_dist,   2)
            target_price = round(entry + target_dist, 2)
        else:
            stop_price   = round(entry + stop_dist,   2)
            target_price = round(entry - target_dist, 2)

        event_type = _best_event_type(conn, c["symbol"])

        try:
            conn.execute("""
                INSERT OR IGNORE INTO signal_outcomes
                (symbol, signal_date, signal, final_score, regime, direction,
                 strategy_bucket, entry_price, atr_at_signal, stop_price, target_price,
                 event_type, outcome, paper_exit)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,'PENDING','PENDING')
            """, (
                c["symbol"], run_date, c["action"], c["final_score"],
                regime_label, c["direction"], c["strategy_bucket"],
                entry, atr, stop_price, target_price, event_type,
            ))
            saved += 1
        except Exception:
            pass

    conn.commit()
    conn.close()
    if verbose:
        print(f"[tracker] recorded {saved} signals for {run_date}")
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# Update outcomes
# ─────────────────────────────────────────────────────────────────────────────

def update_outcomes(verbose: bool = True) -> int:
    conn = get_conn()
    pending = conn.execute("""
        SELECT id, symbol, signal_date, direction, entry_price,
               stop_price, target_price
        FROM signal_outcomes
        WHERE paper_exit = 'PENDING'
        ORDER BY signal_date ASC
    """).fetchall()

    updated = 0
    for row in pending:
        sid       = row["id"]
        symbol    = row["symbol"]
        sig_date  = row["signal_date"]
        direction = row["direction"]
        entry     = float(row["entry_price"])
        stop      = float(row["stop_price"] or 0)
        target    = float(row["target_price"] or 0)

        daily = _all_prices_after(conn, symbol, sig_date, max_days=10)
        if not daily:
            continue

        # Fill t+1 / t+3 / t+5 point-in-time prices (EOD close for P&L)
        t1_price = t1_date = t3_price = t3_date = t5_price = t5_date = None
        if len(daily) >= 1:  t1_price, t1_date = daily[0][0], daily[0][1]
        if len(daily) >= 3:  t3_price, t3_date = daily[2][0], daily[2][1]
        if len(daily) >= 5:  t5_price, t5_date = daily[4][0], daily[4][1]

        t1_pnl = _pnl_pct(entry, t1_price, direction)
        t3_pnl = _pnl_pct(entry, t3_price, direction)
        t5_pnl = _pnl_pct(entry, t5_price, direction)

        # Paper trade exit: use intraday high/low to detect stop/target breach
        # Falls back to close if day_high/day_low not yet populated (older rows)
        paper_exit   = "PENDING"
        paper_pnl    = None
        exit_price   = None

        for close, _, day_high, day_low in daily[:5]:
            hi = day_high if day_high is not None else close
            lo = day_low  if day_low  is not None else close
            if direction == "LONG":
                if stop   and lo <= stop:    paper_exit = "HIT_STOP";   exit_price = stop;   break
                if target and hi >= target:  paper_exit = "HIT_TARGET"; exit_price = target; break
            else:
                if stop   and hi >= stop:    paper_exit = "HIT_STOP";   exit_price = stop;   break
                if target and lo <= target:  paper_exit = "HIT_TARGET"; exit_price = target; break

        # If t+5 data available and still no stop/target hit → T5_EXIT
        if paper_exit == "PENDING" and t5_price is not None:
            paper_exit = "T5_EXIT"
            exit_price = t5_price

        if exit_price:
            paper_pnl = _pnl_pct(entry, exit_price, direction)

        # Resolve MONITOR-level outcome (based on t5 pnl)
        if t5_pnl is not None:
            if   abs(t5_pnl) < 0.5: outcome = "SCRATCH"
            elif t5_pnl > 0:         outcome = "WIN"
            else:                    outcome = "LOSS"
        else:
            outcome = "PENDING"

        conn.execute("""
            UPDATE signal_outcomes SET
                t1_date=?, t1_price=?, t1_pnl_pct=?,
                t3_date=?, t3_price=?, t3_pnl_pct=?,
                t5_date=?, t5_price=?, t5_pnl_pct=?,
                paper_pnl_pct=?, paper_exit=?, outcome=?
            WHERE id=?
        """, (
            t1_date, t1_price, t1_pnl,
            t3_date, t3_price, t3_pnl,
            t5_date, t5_price, t5_pnl,
            paper_pnl, paper_exit, outcome,
            sid,
        ))
        updated += 1

    conn.commit()
    conn.close()
    if verbose and updated:
        print(f"[tracker] updated {updated} pending outcomes")
    return updated


# ─────────────────────────────────────────────────────────────────────────────
# Summary stats (used by app.py and evaluator)
# ─────────────────────────────────────────────────────────────────────────────

def get_outcome_stats(conn) -> dict:
    rows = conn.execute("""
        SELECT outcome, t5_pnl_pct, paper_exit, paper_pnl_pct
        FROM signal_outcomes WHERE outcome != 'PENDING'
    """).fetchall()

    if not rows:
        return {"total": 0}

    total  = len(rows)
    wins   = sum(1 for r in rows if r["outcome"] == "WIN")
    losses = sum(1 for r in rows if r["outcome"] == "LOSS")
    pnls   = [r["t5_pnl_pct"] for r in rows if r["t5_pnl_pct"] is not None]
    avg    = round(sum(pnls) / len(pnls), 2) if pnls else 0.0

    paper_pnls = [r["paper_pnl_pct"] for r in rows if r["paper_pnl_pct"] is not None]
    avg_paper  = round(sum(paper_pnls) / len(paper_pnls), 2) if paper_pnls else 0.0

    return {
        "total":          total,
        "wins":           wins,
        "losses":         losses,
        "win_rate":       round(wins / total * 100, 1),
        "avg_t5_pnl_pct": avg,
        "avg_paper_pnl":  avg_paper,
    }
