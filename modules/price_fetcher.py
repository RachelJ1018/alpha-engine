"""
price_fetcher.py — pulls OHLCV + technicals for watchlist symbols via yfinance
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, date
from modules.db import get_conn

def compute_atr(hist: pd.DataFrame, period=14) -> float:
    """Return 14-day Average True Range (absolute $)."""
    if hist is None or len(hist) < period + 1:
        return None
    high  = hist["High"]
    low   = hist["Low"]
    close = hist["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    return round(float(atr), 4) if pd.notna(atr) else None


def compute_rsi(prices: pd.Series, period=14) -> float:
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, 1e-9)
    rsi   = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 1) if not rsi.empty else None

def fetch_prices(symbols=None, verbose=True):
    conn = get_conn()
    today = date.today().isoformat()

    if symbols is None:
        rows = conn.execute(
            "SELECT symbol FROM watched_symbols WHERE enabled=1"
        ).fetchall()
        symbols = [r["symbol"] for r in rows]

    saved = 0
    failed = []

    for sym in symbols:
        try:
            tk = yf.Ticker(sym)
            hist = tk.history(period="60d", interval="1d", auto_adjust=True)

            if hist.empty or len(hist) < 5:
                failed.append(sym)
                continue

            close   = hist["Close"]
            volume  = hist["Volume"]
            last    = float(close.iloc[-1])
            prev    = float(close.iloc[-2])
            chg_pct  = round((last - prev) / prev * 100, 2)
            day_high = round(float(hist["High"].iloc[-1]), 4)
            day_low  = round(float(hist["Low"].iloc[-1]),  4)

            avg_vol    = float(volume.iloc[:-1].mean())
            today_vol  = float(volume.iloc[-1])
            vol_ratio  = round(today_vol / avg_vol, 2) if avg_vol > 0 else None

            ma20 = round(float(close.rolling(20).mean().iloc[-1]), 2) if len(close) >= 20 else None
            ma50 = round(float(close.rolling(50).mean().iloc[-1]), 2) if len(close) >= 50 else None
            rsi  = compute_rsi(close)
            atr  = compute_atr(hist)
            above_ma20 = 1 if (ma20 and last > ma20) else 0

            info = tk.fast_info
            week_high = round(float(getattr(info, "year_high", 0) or 0), 2)
            week_low  = round(float(getattr(info, "year_low",  0) or 0), 2)
            mktcap    = float(getattr(info, "market_cap", 0) or 0)

            conn.execute("""
                INSERT OR REPLACE INTO price_snapshots
                (symbol, snapshot_date, close_price, change_pct, volume, avg_volume,
                 volume_ratio, rsi_14, ma_20, ma_50, above_ma20, week_high_52, week_low_52,
                 market_cap, atr_14, day_high, day_low)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (sym, today, last, chg_pct, today_vol, avg_vol,
                  vol_ratio, rsi, ma20, ma50, above_ma20, week_high, week_low, mktcap, atr,
                  day_high, day_low))
            saved += 1

        except Exception as e:
            failed.append(sym)
            if verbose:
                print(f"[price] ⚠ {sym}: {e}")

    conn.commit()
    conn.close()

    if verbose:
        print(f"[price] saved {saved}/{len(symbols)} symbols | failed: {failed[:5]}")
    return saved

def get_market_regime(conn) -> dict:
    """Determine today's market regime from SPY."""
    today = date.today().isoformat()
    spy = conn.execute(
        "SELECT * FROM price_snapshots WHERE symbol='SPY' AND snapshot_date=?", (today,)
    ).fetchone()

    if not spy:
        return {"regime": "unknown", "spy_change": 0.0}

    chg  = spy["change_pct"] or 0
    rsi  = spy["rsi_14"] or 50
    vrat = spy["volume_ratio"] or 1

    if chg > 1.0 and rsi < 75:
        regime = "bull"
    elif chg < -1.0 and rsi > 25:
        regime = "bear"
    elif abs(chg) < 0.3 and vrat < 0.8:
        regime = "choppy"
    else:
        regime = "neutral"

    return {"regime": regime, "spy_change": chg, "spy_rsi": rsi}
