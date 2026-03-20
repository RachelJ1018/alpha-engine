"""
options_flow.py — Institutional options flow signals via yfinance.

Computes a boost score (−2 to +3) based on:
  1. Call/Put volume ratio (directional bias of options market)
  2. Volume/OI ratio (unusual activity = institutional bets)

Used by analyzer.py to adjust EventEdge score after news-based scoring.

Notes:
  - Uses nearest expiry in the 7–45 day window (avoids noisy weekly options)
  - Returns 0.0 silently on any data error (non-blocking)
  - ETFs excluded from boost — their CP ratio reflects hedging, not directional bets
  - Results cached per symbol per day to avoid repeated yfinance calls
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ETFs — skip directional options boost (vol is structural/hedging)
_ETF_SYMBOLS = {"SPY", "QQQ", "IWM", "SMH", "GLD", "TLT", "XLF"}

# Cache raw metrics per symbol (direction-independent): {symbol: (date_str, cp_ratio, vol_oi_ratio)}
_cache: dict[str, tuple[str, float, float]] = {}


def score_options_flow(symbol: str, direction: str, current_price: Optional[float] = None) -> float:
    """
    Return an options-flow boost for EventEdge scoring.

    Range: −2.0 (put-heavy, against LONG) to +3.0 (strongly aligned flow).
    Returns 0.0 on any fetch error or missing data.
    """
    if symbol in _ETF_SYMBOLS:
        return 0.0

    today = date.today().isoformat()

    # Cache raw metrics (cp_ratio, vol_oi_ratio) — direction-specific boost computed fresh
    if symbol in _cache and _cache[symbol][0] == today:
        _, cp_ratio, vol_oi_ratio = _cache[symbol]
    else:
        cp_ratio, vol_oi_ratio = _fetch_metrics(symbol, current_price)
        _cache[symbol] = (today, cp_ratio, vol_oi_ratio)

    return _score(direction, cp_ratio, vol_oi_ratio)


def _fetch_metrics(symbol: str, current_price: Optional[float]) -> tuple[float, float]:
    """Return (cp_ratio, vol_oi_ratio). Returns (1.0, 0.0) on any error (neutral)."""
    try:
        import yfinance as yf
        tk = yf.Ticker(symbol)
        expirations = tk.options  # tuple of date strings, nearest first

        if not expirations:
            return 1.0, 0.0

        # Pick nearest expiry in 7–45 day window
        today_dt = date.today()
        selected = None
        for exp in expirations:
            exp_dt = date.fromisoformat(exp)
            days_out = (exp_dt - today_dt).days
            if 7 <= days_out <= 45:
                selected = exp
                break
        if selected is None:
            selected = expirations[0]

        chain = tk.option_chain(selected)
        calls = chain.calls
        puts  = chain.puts

        if calls.empty or puts.empty:
            return 1.0, 0.0

        call_vol = float(calls["volume"].fillna(0).sum())
        put_vol  = float(puts["volume"].fillna(0).sum())
        call_oi  = float(calls["openInterest"].fillna(0).sum())
        put_oi   = float(puts["openInterest"].fillna(0).sum())

        total_vol = call_vol + put_vol
        total_oi  = call_oi  + put_oi

        if total_vol < 100:
            return 1.0, 0.0

        cp_ratio     = call_vol / (put_vol + 1)
        vol_oi_ratio = total_vol / (total_oi + 1) if total_oi > 0 else 0.0

        logger.debug(f"[options] {symbol} cp={cp_ratio:.2f} vol/oi={vol_oi_ratio:.2f}")
        return cp_ratio, vol_oi_ratio

    except Exception as e:
        logger.debug(f"[options] {symbol} fetch failed: {e}")
        return 1.0, 0.0


def _score(direction: str, cp_ratio: float, vol_oi_ratio: float) -> float:
    """
    Convert CP ratio + vol/OI ratio into a boost score.

    CP ratio interpretation:
      > 2.0  : strongly call-heavy (institutional buying calls)
      1.3–2.0: mild call bias
      0.8–1.3: balanced / neutral
      0.5–0.8: mild put bias
      < 0.5  : strongly put-heavy

    Vol/OI ratio:
      > 0.5  : unusual activity (volume exceeds half of open interest)
    """
    boost = 0.0

    if direction == "LONG":
        if   cp_ratio > 2.0:  boost += 2.5
        elif cp_ratio > 1.3:  boost += 1.5
        elif cp_ratio < 0.5:  boost -= 1.5   # put-heavy vs LONG thesis
        elif cp_ratio < 0.8:  boost -= 0.5
    else:  # SHORT
        if   cp_ratio < 0.5:  boost += 2.5
        elif cp_ratio < 0.8:  boost += 1.5
        elif cp_ratio > 2.0:  boost -= 1.5   # call-heavy vs SHORT thesis
        elif cp_ratio > 1.3:  boost -= 0.5

    # Unusual volume adds confidence regardless of direction alignment
    if vol_oi_ratio > 0.5:
        boost += 0.5

    return round(max(-2.0, min(3.0, boost)), 1)
