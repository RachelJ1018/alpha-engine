"""
analyzer.py — v2
Research-first analyzer:
news + price -> layered scores -> candidate -> thesis

Design goals:
- Replace one-shot weighted sum with explainable layered scoring
- Reduce overreliance on naive sentiment and raw volume spike
- Use action labels that fit a research tool:
  ACTIONABLE / WATCHLIST / MONITOR / IGNORE
"""

import json
import os
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from modules.db import get_conn

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY    = os.environ.get("GOOGLE_API_KEY", "")
GROQ_API_KEY      = os.environ.get("GROQ_API_KEY", "")

# Thesis generation config — change these to control cost vs quality
# THESIS_PROVIDER options: "anthropic" | "google" | "groq" | "auto"
#   "auto"      = Anthropic Haiku/Sonnet tiered by action label
#   "google"    = Gemini Flash (requires billing-enabled key)
#   "groq"      = Llama 3.3 70B via Groq (free, no billing needed)
#   "anthropic" = Anthropic only
#   "none"      = Rule-based fallback, no API calls
THESIS_PROVIDER   = os.environ.get("THESIS_PROVIDER", "auto")

# ---------------------------------------------------------------------
# Data-driven weight loader
# ---------------------------------------------------------------------
# If data/weights.json exists (written by weight_optimizer.py --apply),
# its values override the static defaults below.  Delete the file to
# revert to hardcoded defaults at any time.

_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "weights.json")

def _load_weights() -> dict:
    if not os.path.exists(_WEIGHTS_PATH):
        return {}
    try:
        with open(_WEIGHTS_PATH) as f:
            data = json.load(f)
        print(f"[analyzer] loaded data-driven weights from {_WEIGHTS_PATH}")
        return data
    except Exception as e:
        print(f"[analyzer] warning: could not load weights.json ({e}) — using defaults")
        return {}

_LOADED_WEIGHTS = _load_weights()

# ---------------------------------------------------------------------
# Static config
# ---------------------------------------------------------------------

_EVENT_IMPORTANCE_DEFAULTS = {
    "earnings": 1.00,
    "macro": 0.85,
    "ma": 0.85,
    "ai": 0.80,
    "regulation": 0.75,
    "layoff": 0.60,
    "product": 0.55,
    "general": 0.20,
}
# Merge: data-driven overrides win; defaults fill the rest
EVENT_IMPORTANCE = {
    **_EVENT_IMPORTANCE_DEFAULTS,
    **_LOADED_WEIGHTS.get("event_importance", {}),
}

# Discrete strength points used directly in score_event_edge().
# Hard catalysts (earnings/ma/macro) dominate; hype events (ai/product) earn less.
EVENT_STRENGTH = {
    "earnings":   8,
    "ma":         7,
    "macro":      7,
    "regulation": 6,
    "ai":         5,
    "product":    4,
    "layoff":     3,
    "general":    1,
}

# Layer multipliers — applied in compute_final_score().
# Default 1.0 = no change. weight_optimizer.py --apply updates these
# based on which layers actually correlate with t+5 P&L in your DB.
_LAYER_MULTIPLIERS_DEFAULTS = {
    "event_edge_score":    1.0,
    "market_conf_score":   1.0,
    "regime_fit_score":    1.0,
    "relative_opp_score":  1.0,
    "freshness_score":     1.0,
    "risk_penalty_score":  1.0,
}
LAYER_MULTIPLIERS = {
    **_LAYER_MULTIPLIERS_DEFAULTS,
    **_LOADED_WEIGHTS.get("layer_multipliers", {}),
}

SOURCE_TIER = {
    "reuters": 1.00,
    "sec": 1.00,
    "sec filing": 1.00,
    "edgar": 1.00,
    "company ir": 0.95,
    "associated press": 0.90,
    "ap": 0.90,
    "bloomberg": 0.90,
    "wsj": 0.90,
    "financial times": 0.90,
    "marketwatch": 0.70,
    "yahoo finance": 0.55,
    "benzinga": 0.35,
    "unknown": 0.40,
}

def normalize_event_type(article) -> str:
    """
    Normalize noisy event_type labels using title/content hints.
    Keeps analyzer more robust even if upstream news labeling is messy.
    """
    evt = (article["event_type"] or "general").lower()
    title = (article["title"] or "").lower()
    content = (article["content"] or "").lower()
    source = (article["source"] or "").lower()

    text = f"{title} {content}"

    if any(x in text for x in [
        "gdp", "inflation", "cpi", "ppi", "fed", "powell",
        "rates", "rate cut", "rate hike", "treasury yield", "jobs report"
    ]):
        return "macro"

    if any(x in text for x in [
        "acquire", "acquisition", "merger", "buyout", "take private", "deal talks"
    ]):
        return "ma"

    if any(x in text for x in [
        "sec", "investigation", "probe", "lawsuit", "settlement",
        "antitrust", "regulator", "regulatory", "doj", "ftc"
    ]):
        return "regulation"

    if any(x in text for x in [
        "earnings", "eps", "revenue", "guidance", "quarterly results",
        "beat estimate", "miss estimate", "profit outlook"
    ]):
        return "earnings"

    if any(x in text for x in [
        "ai", "artificial intelligence", "model release", "chip launch"
    ]):
        return "ai"

    if any(x in text for x in [
        "launch", "new product", "unveil", "released", "rollout"
    ]):
        return "product"

    if any(x in text for x in [
        "price prediction", "stock prediction", "2026", "2027", "2030",
        "is it a buy", "should you buy", "opinion", "editorial"
    ]):
        return "general"

    if "benzinga" in source and evt == "earnings":
        return "general"

    return evt

# hard caps / thresholds
MIN_CANDIDATE_SCORE = 40.0
MAX_NEWS_ITEMS_PER_SYMBOL = 10

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    return round(max(lo, min(hi, x)), 2)

def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default

def _source_weight(source: Optional[str]) -> float:
    if not source:
        return SOURCE_TIER["unknown"]
    s = source.strip().lower()
    for k, v in SOURCE_TIER.items():
        if k in s:
            return v
    return SOURCE_TIER["unknown"]

def _event_weight(event_type: Optional[str]) -> float:
    return EVENT_IMPORTANCE.get((event_type or "general").lower(), 0.20)

def _hours_since(published_at: Optional[str]) -> Optional[float]:
    if not published_at:
        return None
    try:
        dt = datetime.fromisoformat(str(published_at).replace("Z", "+00:00"))
        now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
        return max((now - dt).total_seconds() / 3600.0, 0.0)
    except Exception:
        return None

def _normalize_sentiment(sent: float) -> float:
    return (sent + 1.0) / 2.0

def _weighted_avg(values: List[float], weights: List[float], default: float = 0.0) -> float:
    if not values or not weights or sum(weights) <= 0:
        return default
    return sum(v * w for v, w in zip(values, weights)) / sum(weights)

# ---------------------------------------------------------------------
# News aggregation
# ---------------------------------------------------------------------

def score_news_bundle(articles, symbol=None) -> dict:
    if not articles:
        return {
            "sentiment": 0.0,
            "event_importance": 0.2,
            "event": 0.2,
            "novelty": 0.5,
            "importance": 0.2,
            "source_quality": 0.4,
            "mixedness": 0.0,
            "best_event_type": "general",
            "count": 0,
            "importances": [],
        }

    def _local_source_weight(source: str) -> float:
        s = (source or "").strip().lower()
        if "reuters" in s:             return 1.00
        if "sec" in s or "edgar" in s: return 1.00
        if "company ir" in s:          return 0.95
        if "associated press" in s or s == "ap": return 0.90
        if "bloomberg" in s or "wsj" in s or "financial times" in s: return 0.90
        if "marketwatch" in s or "cnbc" in s: return 0.75
        if "yahoo finance" in s:       return 0.55
        if "benzinga" in s or "seeking alpha" in s: return 0.35
        return 0.40

    sentiments = []
    novelties = []
    importances = []
    source_weights = []
    article_weights = []
    filtered_event_types = []

    for a in articles:
        sent  = a["sentiment_score"]  if a["sentiment_score"]  is not None else 0.0
        nov   = a["novelty_score"]    if a["novelty_score"]    is not None else 0.5
        imp   = a["importance_score"] if a["importance_score"] is not None else 0.3
        src_w = _local_source_weight(a["source"])

        sentiments.append(float(sent))
        novelties.append(float(nov))
        importances.append(float(imp))
        source_weights.append(float(src_w))

        article_w = max(0.05, float(imp) * 0.6 + float(src_w) * 0.4)
        article_weights.append(article_w)

        evt = normalize_event_type(a)

        try:
            syms = json.loads(a["symbols"] or "[]")
        except Exception:
            syms = []

        if evt == "earnings" and symbol and symbol not in syms:
            continue

        filtered_event_types.append(evt)

    if not filtered_event_types:
        filtered_event_types = ["general"]

    total_w = sum(article_weights)
    weighted_sent  = sum(s * w for s, w in zip(sentiments,     article_weights)) / total_w if total_w > 0 else 0.0
    weighted_nov   = sum(n * w for n, w in zip(novelties,      article_weights)) / total_w if total_w > 0 else 0.5
    weighted_imp   = sum(i * w for i, w in zip(importances,    article_weights)) / total_w if total_w > 0 else 0.3
    weighted_srcq  = sum(s * w for s, w in zip(source_weights, article_weights)) / total_w if total_w > 0 else 0.4

    best_event  = max(filtered_event_types, key=lambda e: EVENT_IMPORTANCE.get(e, 0.2))
    event_score = EVENT_IMPORTANCE.get(best_event, 0.2)

    pos = sum(1 for s in sentiments if s >  0.2)
    neg = sum(1 for s in sentiments if s < -0.2)
    mixedness = 1.0 if pos > 0 and neg > 0 else 0.0

    return {
        "sentiment":        round(weighted_sent,  3),
        "event_importance": round(event_score,    3),
        "event":            round(event_score,    3),
        "novelty":          round(weighted_nov,   3),
        "importance":       round(weighted_imp,   3),
        "source_quality":   round(weighted_srcq,  3),
        "mixedness":        round(mixedness,      3),
        "best_event_type":  best_event,
        "count":            len(articles),
        "importances":      importances,
    }

def is_low_value_title(title: str) -> bool:
    t = (title or "").lower()
    bad_patterns = [
        "here's what", "here's what",
        "has to say about", "think about", "thinks about",
        "maintains a hold rating", "maintains a buy rating", "maintains a sell rating",
        "price prediction", "stock prediction",
        "is it a buy", "should you buy", "top stock to buy",
        "why this stock", "outlier as",
    ]
    return any(p in t for p in bad_patterns)

# ---------------------------------------------------------------------
# Technical / confirmation scoring
# ---------------------------------------------------------------------

def score_technical(price_row: Optional[Any]) -> float:
    if not price_row:
        return 50.0

    score = 50.0
    rsi       = _safe_float(price_row["rsi_14"],      50.0)
    vr        = _safe_float(price_row["volume_ratio"], 1.0)
    chg       = _safe_float(price_row["change_pct"],   0.0)
    above_ma20 = int(price_row["above_ma20"] or 0)
    atr       = _safe_float(price_row["atr_14"],       0.0)
    px        = _safe_float(price_row["close_price"],  0.0)

    if 42 <= rsi <= 63:   score += 12
    elif 35 <= rsi < 42 or 63 < rsi <= 70: score += 5
    elif rsi > 78:        score -= 12
    elif rsi < 25:        score -= 4

    if vr >= 2.0:    score += 4
    elif vr >= 1.3:  score += 2
    elif vr < 0.6:   score -= 6

    if above_ma20:   score += 8
    else:            score -= 5

    if abs(chg) > 5:         score -= 8
    elif 0.5 < chg < 3:      score += 4

    if atr > 0 and px > 0:
        if atr / px * 100 > 6:
            score -= 8

    return _clamp(score, 0, 100)

def score_market_confirmation(price_row: Optional[Any], direction: str, news_scores: Dict[str, Any]) -> float:
    """0-20: confirms price action supports thesis without rewarding overextension."""
    if not price_row:
        return 8.0

    score      = 8.0
    chg        = _safe_float(price_row["change_pct"],   0.0)
    vr         = _safe_float(price_row["volume_ratio"],  1.0)
    rsi        = _safe_float(price_row["rsi_14"],       50.0)
    above_ma20 = int(price_row["above_ma20"] or 0)

    if direction == "LONG":
        if chg > 0:          score += 3
        if above_ma20:       score += 3
        if 1.1 <= vr <= 2.5: score += 3
        elif vr > 3.0:       score += 1
        if rsi > 75:         score -= 3
        if chg > 4.0:        score -= 3
    else:
        if chg < 0:          score += 3
        if not above_ma20:   score += 3
        if 1.1 <= vr <= 2.5: score += 3
        elif vr > 3.0:       score += 1
        if rsi < 25:         score -= 3
        if chg < -4.0:       score -= 3

    if news_scores.get("mixedness", 0) > 0:
        score -= 2

    return _clamp(score, 0, 20)

# ---------------------------------------------------------------------
# Layered scoring
# ---------------------------------------------------------------------

def determine_direction(news_scores: Dict[str, Any], price_row: Optional[Any]) -> str:
    sent = _safe_float(news_scores.get("sentiment"), 0.0)
    if sent > 0.08:  return "LONG"
    if sent < -0.08: return "SHORT"
    chg = _safe_float(price_row["change_pct"], 0.0) if price_row else 0.0
    return "LONG" if chg >= 0 else "SHORT"

def score_event_edge(news_scores: Dict[str, Any], articles: List[Any]) -> float:
    """0-25: how strong is the information edge?"""
    if not articles:
        return 4.0

    source_q  = _safe_float(news_scores.get("source_quality"),    0.4)
    novelty   = _safe_float(news_scores.get("novelty"),            0.5)
    importance = _safe_float(news_scores.get("importance"),        0.3)
    mixedness = _safe_float(news_scores.get("mixedness"),          0.0)
    count     = int(news_scores.get("count", 0))

    best_evt   = news_scores.get("best_event_type", "general")
    evt_points = EVENT_STRENGTH.get(best_evt, 1)
    score      = source_q * 6.0 + evt_points + novelty * 3.0 + importance * 3.0
    if count >= 2: score += 1.5
    if count >= 4: score += 1.0
    score -= mixedness * 3.0

    return _clamp(score, 0, 25)

def score_regime_fit(direction: str, regime: Dict[str, Any], best_event_type: str) -> float:
    """0-15: favor setups aligned with market regime, but don't hard-ban longs in bear."""
    reg = (regime or {}).get("regime", "unknown")

    if reg == "bull":      base = 11.0 if direction == "LONG"  else 6.0
    elif reg == "bear":    base = 10.5 if direction == "SHORT" else 5.5
    elif reg == "choppy":  base = 6.0
    elif reg == "neutral": base = 8.0
    else:                  base = 7.0

    if best_event_type in ("earnings", "ma", "regulation"):
        base += 1.0

    return _clamp(base, 0, 15)

def score_relative_opportunity(symbol: str, price_row: Optional[Any], conn, direct_count: int = 0, direction: str = "LONG") -> float:
    """0-15: catalyst density, RSI positioning, ATR%, and dollar volume."""
    if not price_row:
        return 7.0

    score = 3.0   # base (lower — earned, not given)
    atr   = _safe_float(price_row["atr_14"],      0.0)
    px    = _safe_float(price_row["close_price"],  0.0)

    # Catalyst density (replaces 52-week-high distance)
    if direct_count >= 3:   score += 5
    elif direct_count >= 1: score += 3
    else:                   score += 1

    # Not crowded (RSI) — reward un-extended setups in thesis direction
    rsi = _safe_float(price_row["rsi_14"], 50.0)
    if direction == "LONG":
        if rsi <= 58:   score += 4
        elif rsi <= 68: score += 2
    else:
        if rsi >= 42:   score += 4
        elif rsi >= 32: score += 2

    # Risk/reward via ATR%
    atr_pct = atr / px * 100 if atr > 0 and px > 0 else 4.0
    if atr_pct < 2.0:   score += 4
    elif atr_pct < 3.5: score += 3
    elif atr_pct < 5.0: score += 2
    else:               score += 1

    # Execution quality (dollar volume: avg_volume * price)
    dollar_vol = _safe_float(price_row["avg_volume"], 0) * _safe_float(price_row["close_price"], 0)
    if dollar_vol > 50_000_000:  score += 2
    elif dollar_vol > 10_000_000: score += 1

    return _clamp(score, 0, 15)

def score_freshness(articles: List[Any]) -> float:
    """0-10: reward new information."""
    if not articles:
        return 3.0

    best = 3.0
    for a in articles:
        hrs = _hours_since(a["published_at"])
        nov = _safe_float(a["novelty_score"], 0.5)
        if hrs is None:    age_score = 4.0
        elif hrs <= 6:     age_score = 9.0
        elif hrs <= 24:    age_score = 7.0
        elif hrs <= 48:    age_score = 5.0
        else:              age_score = 2.5
        best = max(best, age_score * 0.6 + nov * 4.0)

    return _clamp(best, 0, 10)

def score_risk_penalty(price_row: Optional[Any], direction: str, news_scores: Dict[str, Any], regime: Dict[str, Any]) -> float:
    """0-15 (subtracted): penalize overextension, volatility, mixed signals, weak macro fit."""
    penalty = 0.0

    if price_row:
        chg = _safe_float(price_row["change_pct"], 0.0)
        rsi = _safe_float(price_row["rsi_14"],     50.0)
        atr = _safe_float(price_row["atr_14"],      0.0)
        px  = _safe_float(price_row["close_price"], 0.0)

        if abs(chg) > 5.0:   penalty += 3.0
        elif abs(chg) > 3.0: penalty += 1.5

        # Extreme RSI in thesis direction
        if direction == "LONG"  and rsi > 75: penalty += 2.5
        if direction == "SHORT" and rsi < 25: penalty += 2.5

        # Counter-trend RSI: shorting overbought / longing oversold
        if direction == "SHORT" and rsi > 70: penalty += 2.5
        if direction == "LONG"  and rsi < 30: penalty += 1.5

        if atr > 0 and px > 0:
            atr_pct = atr / px * 100
            if atr_pct > 7.0:   penalty += 3.0
            elif atr_pct > 5.0: penalty += 1.5

    if _safe_float(news_scores.get("mixedness"), 0.0) > 0:
        penalty += 2.5

    reg = (regime or {}).get("regime", "unknown")
    if reg == "bear"   and direction == "LONG":  penalty += 2.0
    if reg == "bull"   and direction == "SHORT": penalty += 1.5
    if reg == "choppy":                          penalty += 1.5

    return _clamp(penalty, 0, 15)

def map_strategy_bucket(best_event_type: str, direction: str) -> str:
    evt = (best_event_type or "general").lower()
    if evt == "earnings":             return "post_earnings_drift"
    if direction == "SHORT":          return "event_short"
    if evt in ("ai", "product", "ma"): return "sympathy_play"
    return "event_long"

def compute_final_score(
    event_edge_score: float,
    market_conf_score: float,
    regime_fit_score: float,
    relative_opp_score: float,
    freshness_score: float,
    risk_penalty_score: float,
) -> float:
    m = LAYER_MULTIPLIERS
    raw = (
        event_edge_score    * m.get("event_edge_score",   1.0)
        + market_conf_score * m.get("market_conf_score",  1.0)
        + regime_fit_score  * m.get("regime_fit_score",   1.0)
        + relative_opp_score * m.get("relative_opp_score", 1.0)
        + freshness_score   * m.get("freshness_score",    1.0)
        - risk_penalty_score * m.get("risk_penalty_score", 1.0)
    )
    return _clamp(raw, 0, 100)

def determine_action(final_score: float, direction: str, regime: Dict[str, Any]) -> str:
    """ACTIONABLE / WATCHLIST / MONITOR / IGNORE — regime-aware thresholds."""
    reg = (regime or {}).get("regime", "unknown")

    if direction == "LONG":
        actionable = 80 if reg == "bear" else 74 if reg == "bull" else 77
        watch, monitor = 62, 48
    else:
        actionable = 70 if reg == "bear" else 78 if reg == "bull" else 74
        watch, monitor = 60, 48

    if final_score >= actionable: return "ACTIONABLE"
    if final_score >= watch:      return "WATCHLIST"
    if final_score >= monitor:    return "MONITOR"
    return "IGNORE"

# ---------------------------------------------------------------------
# Thesis generation — multi-provider
# ---------------------------------------------------------------------

def _rule_based_thesis(
    symbol: str,
    headlines: List[str],
    price_row: Optional[Any],
    news_scores: Dict[str, Any],
    direction: str,
    action: str,
) -> Dict[str, str]:
    sent  = news_scores.get("sentiment", 0.0)
    price = _safe_float(price_row["close_price"], 0.0) if price_row else 0.0
    ma20  = _safe_float(price_row["ma_20"],  price * 0.97) if price_row else price * 0.97
    atr   = _safe_float(price_row["atr_14"], max(price * 0.02, 1.0)) if price_row else max(price * 0.02, 1.0)

    if direction == "LONG":
        stop   = max(price - 1.2 * atr, 0.01)
        target = price + 2.0 * atr
    else:
        stop   = price + 1.2 * atr
        target = max(price - 2.0 * atr, 0.01)

    tone = "positive" if sent >= 0 else "negative"
    top  = headlines[0] if headlines else None

    thesis = (
        f"{top}. Headline tone is {tone}, current regime keeps this in {action.lower()} status."
        if top else
        f"No symbol-specific catalyst for {symbol}. View driven by market context and price action."
    )
    return {
        "thesis":         thesis,
        "entry_note":     f"Prefer confirmation near ${price:.2f}; avoid chasing.",
        "stop_loss_note": f"Risk below/above ${stop:.2f}; MA20 reference ${ma20:.2f}.",
        "target_note":    f"Initial target near ${target:.2f} over 3-5 trading days.",
        "risk_note":      "Thesis fails if regime flips sharply or follow-through absent.",
        "direction":      direction,
        "conviction":     "MEDIUM" if action in ("ACTIONABLE", "WATCHLIST") else "LOW",
    }


def _build_thesis_prompt(
    symbol: str,
    company_name: str,
    news_headlines: List[str],
    price_row: Optional[Any],
    news_scores: Dict[str, Any],
    component_scores: Dict[str, float],
    regime: Dict[str, Any],
    direction: str,
    action: str,
) -> tuple:
    """Returns (system_prompt, user_prompt) for any LLM provider."""
    price = _safe_float(price_row["close_price"], 0.0) if price_row else 0.0
    chg   = _safe_float(price_row["change_pct"],  0.0) if price_row else 0.0
    rsi   = _safe_float(price_row["rsi_14"],      50.0) if price_row else None
    vr    = _safe_float(price_row["volume_ratio"], 1.0) if price_row else None
    ma20  = _safe_float(price_row["ma_20"],        0.0) if price_row else None
    ma50  = _safe_float(price_row["ma_50"],        0.0) if price_row else None

    headlines_str = "\n".join(f"- {h}" for h in news_headlines[:5]) or "- No major symbol-specific headlines"
    regime_str    = regime.get("regime", "neutral").upper()
    spy_chg       = regime.get("spy_change", 0.0)

    system = """You are a disciplined short-term equity research assistant.
Return ONLY valid JSON — no markdown, no extra text.
Schema:
{
  "thesis": "2 sentences max. Must reference the direction explicitly (LONG or SHORT) and the key catalyst.",
  "entry_note": "specific but conservative entry guidance",
  "stop_loss_note": "specific invalidation or stop guidance",
  "target_note": "specific initial target / timeframe",
  "risk_note": "single biggest risk to the thesis",
  "direction": "LONG or SHORT",
  "conviction": "HIGH / MEDIUM / LOW"
}
Keep it practical and do not invent unavailable facts."""

    user = f"""Generate a concise trade research thesis for {symbol} ({company_name}) for TODAY.

IMPORTANT: The suggested direction is {direction}. Justify WHY {direction}, not the opposite.

Market regime: {regime_str} | SPY: {spy_chg:+.2f}%
Price: ${price:.2f} ({chg:+.2f}%) | RSI: {rsi} | Vol: {vr}x | MA20: {ma20} | MA50: {ma50}

Top headlines:
{headlines_str}

Signals: event={news_scores.get('best_event_type')} | sentiment={news_scores.get('sentiment'):.2f} | novelty={news_scores.get('novelty'):.2f}

Scores: EventEdge={component_scores['event_edge_score']}/25 | MarketConf={component_scores['market_conf_score']}/20 | RegimeFit={component_scores['regime_fit_score']}/15 | Final={component_scores['final_score']}/100 | Action={action}

Respond in JSON only."""

    return system, user


def _call_anthropic(system: str, user: str, action: str) -> Dict[str, Any]:
    """Sonnet for ACTIONABLE, Haiku for everything else."""
    import requests
    model = "claude-sonnet-4-6" if action == "ACTIONABLE" else "claude-haiku-4-5-20251001"
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
        },
        json={
            "model": model,
            "max_tokens": 800,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        },
        timeout=30,
    )
    d = r.json()
    if r.status_code != 200:
        err = d.get("error", {})
        raise RuntimeError(f"Anthropic {r.status_code}: {err.get('message', str(d))}")
    text = "".join(b["text"] for b in (d.get("content") or []) if b.get("type") == "text")
    return json.loads(text.replace("```json", "").replace("```", "").strip())


def _repair_truncated_json(text: str, direction: str) -> Dict[str, Any]:
    """Salvage a truncated JSON response by extracting completed fields via regex."""
    import re
    result = {}
    for match in re.finditer(r'"(\w+)"\s*:\s*"((?:[^"\\]|\\.)*)"', text):
        result[match.group(1)] = match.group(2)
    fallbacks = {
        "thesis":        "Analysis truncated — see layered scores for context.",
        "entry_note":    "Wait for price confirmation before entry.",
        "stop_loss_note": "Use ATR-based stop from risk engine.",
        "target_note":   "Target per 2R calculation in report.",
        "risk_note":     "Thesis incomplete due to response truncation.",
        "direction":     direction,
        "conviction":    "LOW",
    }
    for k, v in fallbacks.items():
        if k not in result:
            result[k] = v
    result["_truncated"] = True
    return result


def _call_google(system: str, user: str, direction: str = "LONG") -> Dict[str, Any]:
    """Call Google Gemini Flash. Falls back to JSON repair if response is truncated."""
    import requests
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-3-flash-preview:generateContent?key={GOOGLE_API_KEY}"
    )
    payload = {
        "contents": [{"parts": [{"text": f"{system}\n\n{user}"}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1800},
    }
    r = requests.post(url, json=payload, timeout=30)
    d = r.json()
    if r.status_code != 200:
        err = d.get("error", {})
        raise RuntimeError(f"Gemini HTTP {r.status_code}: {err.get('message', d)}")
    text = d["candidates"][0]["content"]["parts"][0]["text"]
    cleaned = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"[analyzer] Gemini response truncated — attempting repair")
        return _repair_truncated_json(cleaned, direction)


def _call_groq(system: str, user: str) -> Dict[str, Any]:
    """Call Groq — Llama 3.3 70B. Free tier: 14,400 req/day."""
    import requests
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}",
        },
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "temperature": 0.3,
            "max_tokens": 800,
        },
        timeout=30,
    )
    d = r.json()
    if r.status_code != 200:
        err = d.get("error", {})
        raise RuntimeError(f"Groq HTTP {r.status_code}: {err.get('message', d)}")
    text = d["choices"][0]["message"]["content"]
    return json.loads(text.replace("```json", "").replace("```", "").strip())


def call_claude_for_thesis(
    symbol: str,
    company_name: str,
    news_headlines: List[str],
    price_row: Optional[Any],
    news_scores: Dict[str, Any],
    component_scores: Dict[str, float],
    regime: Dict[str, Any],
    direction: str,
    action: str,
) -> Dict[str, Any]:
    """
    Multi-provider thesis generator.
    Provider chosen by THESIS_PROVIDER env var (default: "auto").
      auto      — Anthropic Haiku/Sonnet; falls back to Groq → Gemini → rule-based
      anthropic — Always Anthropic
      google    — Always Gemini Flash
      groq      — Always Groq / Llama 3.3 70B
      none      — Skip API, use rule-based fallback
    """
    provider = THESIS_PROVIDER.lower()

    if provider == "none" or (
        provider == "auto"
        and not ANTHROPIC_API_KEY
        and not GOOGLE_API_KEY
        and not GROQ_API_KEY
    ):
        return _rule_based_thesis(symbol, news_headlines, price_row, news_scores, direction, action)

    system, user = _build_thesis_prompt(
        symbol, company_name, news_headlines, price_row,
        news_scores, component_scores, regime, direction, action,
    )

    try:
        if provider in ("anthropic", "auto") and ANTHROPIC_API_KEY:
            out = _call_anthropic(system, user, action)
            model_used = "haiku" if action != "ACTIONABLE" else "sonnet"

        elif provider == "google" and GOOGLE_API_KEY:
            out = _call_google(system, user, direction)
            model_used = "gemini-flash"

        elif provider == "groq" and GROQ_API_KEY:
            out = _call_groq(system, user)
            model_used = "llama-3.3-70b"

        elif provider == "auto" and GROQ_API_KEY:
            out = _call_groq(system, user)
            model_used = "llama-3.3-70b"

        elif provider == "auto" and GOOGLE_API_KEY:
            out = _call_google(system, user, direction)   # FIX: pass direction
            model_used = "gemini-flash"

        else:
            return _rule_based_thesis(symbol, news_headlines, price_row, news_scores, direction, action)

        if "direction" not in out:
            out["direction"] = direction
        print(f"[analyzer] thesis for {symbol} via {model_used} ({action})")
        return out

    except Exception as e:
        print(f"[analyzer] thesis API error for {symbol} ({provider}): {e}")
        # Fallback chain: Groq → Gemini → rule-based
        for fallback_name, fallback_fn, fallback_key in [
            ("groq",   lambda: _call_groq(system, user),              GROQ_API_KEY),
            ("gemini", lambda: _call_google(system, user, direction),  GOOGLE_API_KEY),
        ]:
            if fallback_key and provider != fallback_name:
                try:
                    out = fallback_fn()
                    if "direction" not in out:
                        out["direction"] = direction
                    print(f"[analyzer] fallback to {fallback_name} for {symbol}")
                    return out
                except Exception as e2:
                    print(f"[analyzer] {fallback_name} fallback failed for {symbol}: {e2}")
        return _rule_based_thesis(symbol, news_headlines, price_row, news_scores, direction, action)

# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------

def run_analysis(regime: Dict[str, Any], verbose: bool = True) -> int:
    conn = get_conn()
    today = date.today().isoformat()

    symbols = [
        r["symbol"]
        for r in conn.execute(
            "SELECT symbol FROM watched_symbols WHERE enabled=1 ORDER BY priority, symbol"
        ).fetchall()
    ]

    candidates_created = 0

    for sym in symbols:
        price_row = conn.execute(
            "SELECT * FROM price_snapshots WHERE symbol=? AND snapshot_date=?",
            (sym, today),
        ).fetchone()

        articles = conn.execute("""
            SELECT * FROM news_articles
            WHERE symbols LIKE ?
              AND published_at >= datetime('now', '-48 hours')
              AND LOWER(source) NOT LIKE '%benzinga%'
              AND LOWER(source) NOT LIKE '%seeking alpha%'
              AND LOWER(title) NOT LIKE '%price prediction%'
              AND LOWER(title) NOT LIKE '%stock prediction%'
              AND LOWER(title) NOT LIKE '%should you buy%'
            ORDER BY importance_score DESC, published_at DESC
            LIMIT 10
        """, (f'%"{sym}"%',)).fetchall()

        articles = [a for a in articles if not is_low_value_title(a["title"])]

        macro_news = conn.execute("""
            SELECT * FROM news_articles
            WHERE published_at >= datetime('now', '-24 hours')
              AND importance_score >= 0.80
              AND LOWER(source) NOT LIKE '%benzinga%'
              AND LOWER(source) NOT LIKE '%seeking alpha%'
              AND (
                    LOWER(title) LIKE '%gdp%' OR
                    LOWER(title) LIKE '%inflation%' OR
                    LOWER(title) LIKE '%cpi%' OR
                    LOWER(title) LIKE '%ppi%' OR
                    LOWER(title) LIKE '%fed%' OR
                    LOWER(title) LIKE '%powell%' OR
                    LOWER(title) LIKE '%rates%' OR
                    event_type = 'macro'
                  )
            ORDER BY importance_score DESC, published_at DESC
            LIMIT 2
        """).fetchall()

        all_articles = list(articles)
        if not all_articles:
            all_articles += list(macro_news)

        if not price_row and not all_articles:
            continue

        symbol_specific_count = len(articles)
        is_index       = sym in ("SPY", "QQQ")
        has_symbol_news = symbol_specific_count > 0

        news_scores       = score_news_bundle(all_articles, symbol=sym)
        direction         = determine_direction(news_scores, price_row)
        technical_score   = score_technical(price_row)

        event_edge_score    = score_event_edge(news_scores, all_articles)
        market_conf_score   = score_market_confirmation(price_row, direction, news_scores)
        regime_fit_score    = score_regime_fit(direction, regime, news_scores["best_event_type"])
        relative_opp_score  = score_relative_opportunity(sym, price_row, conn, direct_count=symbol_specific_count, direction=direction)
        freshness_score     = score_freshness(all_articles)
        risk_penalty_score  = score_risk_penalty(price_row, direction, news_scores, regime)

        final_score = compute_final_score(
            event_edge_score=event_edge_score,
            market_conf_score=market_conf_score,
            regime_fit_score=regime_fit_score,
            relative_opp_score=relative_opp_score,
            freshness_score=freshness_score,
            risk_penalty_score=risk_penalty_score,
        )

        if not has_symbol_news:
            if is_index:
                event_edge_score = min(event_edge_score, 12.0)
                freshness_score  = min(freshness_score,   6.0)
            else:
                event_edge_score = min(event_edge_score,  8.0)
                freshness_score  = min(freshness_score,   5.0)
            final_score = compute_final_score(
                event_edge_score=event_edge_score,
                market_conf_score=market_conf_score,
                regime_fit_score=regime_fit_score,
                relative_opp_score=relative_opp_score,
                freshness_score=freshness_score,
                risk_penalty_score=risk_penalty_score,
            )
            if not is_index:
                final_score = min(final_score, 59.0)

        low_value = any(is_low_value_title(a["title"]) for a in articles)

        if not has_symbol_news and not is_index:
            strategy_bucket = "macro_watch"
        elif low_value:
            strategy_bucket = "opinion_watch"
        else:
            strategy_bucket = map_strategy_bucket(news_scores["best_event_type"], direction)

        if low_value:
            event_edge_score = min(event_edge_score, 9.0)
            freshness_score  = min(freshness_score,  5.0)

        final_score = compute_final_score(
            event_edge_score=event_edge_score,
            market_conf_score=market_conf_score,
            regime_fit_score=regime_fit_score,
            relative_opp_score=relative_opp_score,
            freshness_score=freshness_score,
            risk_penalty_score=risk_penalty_score,
        )

        action = determine_action(final_score, direction, regime)
        if not has_symbol_news and not is_index and action in ("ACTIONABLE", "WATCHLIST"):
            action = "MONITOR"
        if low_value and action == "ACTIONABLE":
            action = "WATCHLIST"

        headlines    = [a["title"] for a in articles[:5]]
        company_name = sym

        component_scores = {
            "event_edge_score":   event_edge_score,
            "market_conf_score":  market_conf_score,
            "regime_fit_score":   regime_fit_score,
            "relative_opp_score": relative_opp_score,
            "freshness_score":    freshness_score,
            "risk_penalty_score": risk_penalty_score,
            "final_score":        final_score,
        }

        thesis_data = call_claude_for_thesis(
            symbol=sym,
            company_name=company_name,
            news_headlines=headlines,
            price_row=price_row,
            news_scores=news_scores,
            component_scores=component_scores,
            regime=regime,
            direction=direction,
            action=action,
        )

        direction_final = thesis_data.get("direction", direction)
        news_ids = json.dumps([a["id"] for a in articles[:5]])

        conn.execute("""
            INSERT OR REPLACE INTO trade_candidates
            (
                run_date, symbol, company_name, direction, final_score,
                event_score, sentiment_score, technical_score,
                thesis, entry_note, stop_loss_note, target_note, risk_note,
                action, news_ids,
                event_edge_score, market_conf_score, regime_fit_score,
                relative_opp_score, freshness_score, risk_penalty_score,
                strategy_bucket
            )
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                today, sym, company_name, direction_final, final_score,
                round(news_scores.get("event_importance", news_scores.get("event", 0.2)) * 100, 1),
                round(_normalize_sentiment(news_scores["sentiment"]) * 100, 1),
                technical_score,
                thesis_data.get("thesis", ""),
                thesis_data.get("entry_note", ""),
                thesis_data.get("stop_loss_note", ""),
                thesis_data.get("target_note", ""),
                thesis_data.get("risk_note", ""),
                action, news_ids,
                event_edge_score, market_conf_score, regime_fit_score,
                relative_opp_score, freshness_score, risk_penalty_score,
                strategy_bucket,
            ),
        )
        candidates_created += 1

        if verbose:
            print(
                f"[analyze] {sym:6s} score={final_score:5.1f} "
                f"action={action:10s} dir={direction_final:5s} bucket={strategy_bucket}"
            )

    conn.commit()
    conn.close()
    return candidates_created
