"""
price_fetcher.py — OHLCV + technicals for watchlist symbols

Primary source : yfinance
Supplemental   : Longbridge (PE, turnover_rate, volume_ratio refinement)
                 Only triggered when env vars are present AND yfinance data
                 is incomplete or missing supplemental fields.

DataQualityStatus per symbol:
    data_quality = GOOD    — all critical fields present from primary source
                 = PARTIAL — critical OHLCV present, some supplemental missing
                 = MISSING — could not get usable OHLCV
    price_source = yfinance | longbridge | yfinance+longbridge | none
"""
from __future__ import annotations

import os
from datetime import datetime, date
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf

from modules.db import get_conn


# ── ATR / RSI helpers ─────────────────────────────────────────────────────────

def compute_atr(hist: pd.DataFrame, period: int = 14) -> Optional[float]:
    if hist is None or len(hist) < period + 1:
        return None
    high = hist["High"]
    low  = hist["Low"]
    close = hist["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    return round(float(atr), 4) if pd.notna(atr) else None


def compute_rsi(prices: pd.Series, period: int = 14) -> Optional[float]:
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, 1e-9)
    rsi   = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 1) if not rsi.empty else None


# ── yfinance fetch ────────────────────────────────────────────────────────────

def _fetch_yfinance(sym: str) -> Optional[Dict[str, Any]]:
    """
    Returns a dict of price fields, or None if unusable.
    Fields: close, prev_close, change_pct, volume, avg_volume, volume_ratio,
            day_high, day_low, ma20, ma50, rsi, atr, above_ma20,
            week_high, week_low, market_cap, pe_ratio
    """
    try:
        tk   = yf.Ticker(sym)
        hist = tk.history(period="60d", interval="1d", auto_adjust=True)
        if hist.empty or len(hist) < 5:
            return None

        close  = hist["Close"]
        volume = hist["Volume"]
        last   = float(close.iloc[-1])
        prev   = float(close.iloc[-2])

        avg_vol   = float(volume.iloc[:-1].mean())
        today_vol = float(volume.iloc[-1])
        vol_ratio = round(today_vol / avg_vol, 2) if avg_vol > 0 else None

        ma20 = round(float(close.rolling(20).mean().iloc[-1]), 2) if len(close) >= 20 else None
        ma50 = round(float(close.rolling(50).mean().iloc[-1]), 2) if len(close) >= 50 else None

        info      = tk.fast_info
        week_high = round(float(getattr(info, "year_high", 0) or 0), 2)
        week_low  = round(float(getattr(info, "year_low",  0) or 0), 2)
        mktcap    = float(getattr(info, "market_cap", 0) or 0)

        # PE from fast_info (not always populated)
        pe = None
        try:
            pe_raw = getattr(info, "pe_ratio", None)
            if pe_raw and pe_raw > 0:
                pe = round(float(pe_raw), 2)
        except Exception:
            pass

        return {
            "close":       round(last, 4),
            "change_pct":  round((last - prev) / prev * 100, 2),
            "volume":      today_vol,
            "avg_volume":  avg_vol,
            "volume_ratio": vol_ratio,
            "day_high":    round(float(hist["High"].iloc[-1]), 4),
            "day_low":     round(float(hist["Low"].iloc[-1]),  4),
            "ma20":        ma20,
            "ma50":        ma50,
            "rsi":         compute_rsi(close),
            "atr":         compute_atr(hist),
            "above_ma20":  1 if (ma20 and last > ma20) else 0,
            "week_high":   week_high,
            "week_low":    week_low,
            "market_cap":  mktcap,
            "pe_ratio":    pe,
            "turnover_rate": None,   # yfinance doesn't provide this
            "source":      "yfinance",
        }
    except Exception:
        return None


# ── Longbridge supplement ─────────────────────────────────────────────────────

_LB_CTX = None   # module-level cache so we only initialise once per run

def _get_longbridge_ctx():
    """Return a Longbridge QuoteContext, or None if not configured / unavailable."""
    global _LB_CTX
    if _LB_CTX is not None:
        return _LB_CTX

    app_key    = os.environ.get("LONGBRIDGE_APP_KEY")
    app_secret = os.environ.get("LONGBRIDGE_APP_SECRET")
    access_tok = os.environ.get("LONGBRIDGE_ACCESS_TOKEN")
    if not (app_key and app_secret and access_tok):
        return None

    try:
        from longport.openapi import QuoteContext, Config          # type: ignore
        cfg    = Config(app_key=app_key, app_secret=app_secret,
                        access_token=access_tok)
        _LB_CTX = QuoteContext(cfg)
        return _LB_CTX
    except Exception as e:
        print(f"[price] Longbridge init failed: {e}")
        return None


def _fetch_longbridge_supplement(sym: str, ctx) -> Dict[str, Any]:
    """
    Pull supplemental fields from Longbridge for a US-listed symbol.
    Returns a (possibly empty) dict with any of: pe_ratio, turnover_rate, volume_ratio.

    Longbridge notation: "<SYM>.US"

    turnover_rate unit: Longbridge returns a decimal fraction (e.g. 0.0234 = 2.34%).
    We store it as percentage (×100), so DB value of 2.34 means 2.34% daily turnover.

    volume_ratio (量比): ratio of current session volume to average — same scale as
    yfinance-computed volume_ratio, so they are directly comparable.
    """
    out: Dict[str, Any] = {}
    try:
        lb_sym = f"{sym}.US"
        quotes = ctx.quote([lb_sym])
        if not quotes:
            return out
        q = quotes[0]

        # volume_ratio (量比) — highest priority supplemental field
        vr = getattr(q, "volume_ratio", None)
        if vr is not None:
            out["volume_ratio"] = round(float(vr), 2)

        # turnover_rate (换手率) — stored as % (multiply Longbridge decimal × 100)
        tr = getattr(q, "turnover_rate", None)
        if tr is not None:
            out["turnover_rate"] = round(float(tr) * 100, 4)

        # PE ratio
        pe = getattr(q, "pe", None)
        if pe is not None and float(pe) > 0:
            out["pe_ratio"] = round(float(pe), 2)

    except Exception:
        pass
    return out


# ── Data quality classification ───────────────────────────────────────────────

# Tier 1 — required for any usable signal (PARTIAL if any missing)
_CRITICAL_TIER1 = ("close", "change_pct", "volume")
# Tier 2 — required for full scoring; GOOD requires all tier1 + tier2 present
_CRITICAL_TIER2 = ("volume_ratio", "rsi", "atr")

def _classify_quality(row: Optional[Dict[str, Any]]) -> str:
    """
    GOOD    — all tier-1 + tier-2 critical fields present.
              Tier: ACTIONABLE / WATCHLIST / MONITOR / IGNORE all eligible.
    PARTIAL — tier-1 present (close, change_pct, volume) but ≥1 tier-2 missing.
              volume_ratio is the most impactful tier-2 field; its absence means
              MarketConf and ACTIONABLE edge conditions cannot fire reliably.
              Tier cap: WATCHLIST (cannot be ACTIONABLE).
    MISSING — close or change_pct absent; data is not usable for scoring.
              Tier cap: IGNORE (symbol skipped entirely).
    """
    if row is None:
        return "MISSING"
    t1_missing = [f for f in _CRITICAL_TIER1 if row.get(f) is None]
    if t1_missing:
        return "MISSING"
    t2_missing = [f for f in _CRITICAL_TIER2 if row.get(f) is None]
    if t2_missing:
        return "PARTIAL"
    return "GOOD"


def _quality_fields(row: Optional[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Returns {"missing": [...], "supplemented": [...]} for logging / UI.
    'supplemented' = fields that were None after yfinance but filled by Longbridge.
    We detect this by comparing row["source"] and whether lb_fields are present.
    """
    if row is None:
        return {"missing": list(_CRITICAL_TIER1) + list(_CRITICAL_TIER2), "supplemented": []}
    all_tracked = list(_CRITICAL_TIER1) + list(_CRITICAL_TIER2) + ["pe_ratio", "turnover_rate"]
    missing      = [f for f in all_tracked if row.get(f) is None]
    supplemented = row.get("_supplemented", [])
    return {"missing": missing, "supplemented": supplemented}


# ── Main entry point ──────────────────────────────────────────────────────────

def fetch_prices(symbols: Optional[List[str]] = None, verbose: bool = True) -> Dict[str, Any]:
    """
    Fetch prices for all enabled watchlist symbols.
    Returns a DataQualityStatus summary dict:
      {
        "saved":   int,
        "failed":  [sym, ...],
        "quality": {"GOOD": n, "PARTIAL": n, "MISSING": n},
        "sources": {"yfinance": n, "yfinance+longbridge": n, "longbridge": n, "none": n},
      }
    """
    conn  = get_conn()
    today = date.today().isoformat()

    if symbols is None:
        rows    = conn.execute("SELECT symbol FROM watched_symbols WHERE enabled=1").fetchall()
        symbols = [r["symbol"] for r in rows]

    lb_ctx   = _get_longbridge_ctx()
    lb_available = lb_ctx is not None
    if verbose and lb_available:
        print("[price] Longbridge supplement: available")

    saved   = 0
    failed  = []
    quality_counts  = {"GOOD": 0, "PARTIAL": 0, "MISSING": 0}
    source_counts   = {"yfinance": 0, "yfinance+longbridge": 0, "longbridge": 0, "none": 0}
    sym_quality: Dict[str, Dict] = {}   # sym → {quality, missing_fields, supplemented_fields}

    for sym in symbols:
        row = _fetch_yfinance(sym)
        lb_used = False

        # ── Longbridge supplement ────────────────────────────────────────────
        # Priority order: volume_ratio first, then pe_ratio, turnover_rate.
        # Trigger when Longbridge available AND yfinance failed OR missing any of these.
        if lb_available:
            yf_failed     = row is None
            missing_suppl = row is not None and (
                row.get("volume_ratio")  is None or   # highest priority — affects tier gate
                row.get("pe_ratio")      is None or
                row.get("turnover_rate") is None
            )
            if yf_failed or missing_suppl:
                lb_data = _fetch_longbridge_supplement(sym, lb_ctx)
                if lb_data:
                    lb_used = True
                    if row is None:
                        row = {}
                    supplemented = []
                    for k, v in lb_data.items():
                        if row.get(k) is None:
                            row[k] = v
                            supplemented.append(k)
                    row["_supplemented"] = supplemented

        # ── Quality tag ─────────────────────────────────────────────────────
        quality = _classify_quality(row)
        quality_counts[quality] += 1
        qf = _quality_fields(row)
        sym_quality[sym] = {"quality": quality, **qf}

        if quality == "MISSING":
            failed.append(sym)
            source_counts["none"] += 1
            if verbose:
                print(f"[price] ✗ {sym}: MISSING — {qf['missing']}")
            continue

        # ── Source tag ──────────────────────────────────────────────────────
        base_source = row.get("source", "yfinance")
        if lb_used and base_source == "yfinance":
            price_source = "yfinance+longbridge"
        elif lb_used:
            price_source = "longbridge"
        else:
            price_source = base_source
        source_counts[price_source] = source_counts.get(price_source, 0) + 1

        # ── Persist ─────────────────────────────────────────────────────────
        try:
            conn.execute("""
                INSERT OR REPLACE INTO price_snapshots
                (symbol, snapshot_date, close_price, change_pct, volume, avg_volume,
                 volume_ratio, rsi_14, ma_20, ma_50, above_ma20, week_high_52, week_low_52,
                 market_cap, atr_14, day_high, day_low,
                 price_source, data_quality, pe_ratio, turnover_rate)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                sym, today,
                row.get("close"),        row.get("change_pct"),
                row.get("volume"),       row.get("avg_volume"),
                row.get("volume_ratio"), row.get("rsi"),
                row.get("ma20"),         row.get("ma50"),
                row.get("above_ma20"),
                row.get("week_high"),    row.get("week_low"),
                row.get("market_cap"),   row.get("atr"),
                row.get("day_high"),     row.get("day_low"),
                price_source, quality,
                row.get("pe_ratio"),     row.get("turnover_rate"),
            ))
            saved += 1
            if verbose and quality == "PARTIAL":
                print(
                    f"[price] ⚠ {sym}: PARTIAL — missing={qf['missing']} "
                    f"supplemented={qf['supplemented']} source={price_source}"
                )
        except Exception as e:
            failed.append(sym)
            if verbose:
                print(f"[price] ✗ {sym}: DB write error: {e}")

    conn.commit()

    # ── Post-save sanity checks ──────────────────────────────────────────────
    if verbose:
        stale = conn.execute("""
            SELECT a.symbol FROM price_snapshots a
            JOIN price_snapshots b ON a.symbol = b.symbol
            WHERE a.snapshot_date = ? AND b.snapshot_date < ?
              AND a.close_price = b.close_price AND a.close_price IS NOT NULL
            ORDER BY b.snapshot_date DESC
        """, (today, today)).fetchall()
        stale_syms = list({r["symbol"] for r in stale})
        if stale_syms:
            print(f"[price] ⚠ STALE (price unchanged from prev day): {stale_syms}")

        nulls = conn.execute("""
            SELECT symbol FROM price_snapshots
            WHERE snapshot_date = ? AND (close_price IS NULL OR change_pct IS NULL)
        """, (today,)).fetchall()
        if nulls:
            print(f"[price] ⚠ NULL close/change_pct: {[r['symbol'] for r in nulls]}")

    conn.close()

    status = {
        "saved":       saved,
        "failed":      failed,
        "quality":     quality_counts,
        "sources":     source_counts,
        "sym_quality": sym_quality,   # {sym: {quality, missing_fields, supplemented_fields}}
    }

    if verbose:
        print(
            f"[price] {saved}/{len(symbols)} saved | "
            f"quality: {quality_counts} | "
            f"sources: {source_counts}"
        )
        if failed:
            print(f"[price] failed: {failed[:10]}")

    return status


# ── Market regime (unchanged) ─────────────────────────────────────────────────

def get_market_regime(conn) -> dict:
    """Determine today's market regime from SPY."""
    today = date.today().isoformat()
    spy   = conn.execute(
        "SELECT * FROM price_snapshots WHERE symbol='SPY' AND snapshot_date=?", (today,)
    ).fetchone()

    if not spy:
        return {"regime": "unknown", "spy_change": 0.0}

    chg  = spy["change_pct"]    or 0
    rsi  = spy["rsi_14"]        or 50
    vrat = spy["volume_ratio"]  or 1

    if chg > 1.0 and rsi < 75:
        regime = "bull"
    elif chg < -1.0 and rsi > 25:
        regime = "bear"
    elif abs(chg) < 0.3 and vrat < 0.8:
        regime = "choppy"
    else:
        regime = "neutral"

    return {"regime": regime, "spy_change": chg, "spy_rsi": rsi}
