"""
multi_agent_thesis.py — Alpha Engine Multi-Agent Thesis Generator
=================================================================
Drop-in replacement for the single-LLM thesis call in analyzer.py.

Architecture: 4 specialist agents → 1 synthesis agent
  1. TechnicalAgent    — price action, MA, RSI, ATR, volume
  2. NewsAgent         — event quality, sentiment, novelty, catalyst type
  3. RiskAgent         — downside scenarios, position sizing, stop logic
  4. SynthesisAgent    — reads all 3 reports, writes final thesis + conviction

Usage in analyzer.py:
    from modules.multi_agent_thesis import generate_multi_agent_thesis

    thesis = generate_multi_agent_thesis(
        symbol=symbol,
        direction=direction,           # "LONG" | "SHORT"
        score_components=score_dict,   # EventEdge, MarketConf, RegimeFit, RelOpp, RiskPenalty
        price_data=price_row,          # dict from price_snapshots
        news_items=news_list,          # list of dicts from news_articles
        market_regime=regime,          # "bull" | "bear" | "neutral" | "choppy"
        action_label=label,            # "ACTIONABLE" | "WATCHLIST" | "MONITOR"
        provider=provider,             # "anthropic" | "groq" | "google" | "auto"
    )
    # Returns: ThesisResult(summary, technical, news, risk, conviction, raw_agents)
"""

from __future__ import annotations

import json
import os
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────

@dataclass
class AgentReport:
    agent_name: str
    content: str
    tokens_used: int = 0
    provider_used: str = ""
    error: Optional[str] = None


@dataclass
class ThesisResult:
    """Full multi-agent thesis output."""
    summary: str                          # Final 2-3 sentence thesis (shown in report)
    conviction: str                       # "HIGH" | "MEDIUM" | "LOW"
    technical_report: str                 # TechnicalAgent output
    news_report: str                      # NewsAgent output
    risk_report: str                      # RiskAgent output
    synthesis_report: str                 # Full synthesis narrative
    agents_used: list = field(default_factory=list)
    provider_used: str = ""
    fallback: bool = False                # True if rule-based fallback was used

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "conviction": self.conviction,
            "technical_report": self.technical_report,
            "news_report": self.news_report,
            "risk_report": self.risk_report,
            "synthesis_report": self.synthesis_report,
            "agents_used": self.agents_used,
            "provider_used": self.provider_used,
            "fallback": self.fallback,
        }


# ─────────────────────────────────────────────
# Agent prompts
# ─────────────────────────────────────────────

def _technical_prompt(symbol: str, direction: str, price_data: dict, regime: str) -> str:
    close = price_data.get("close_price", "N/A")
    ma20 = price_data.get("ma_20", "N/A")
    ma50 = price_data.get("ma_50", "N/A")
    rsi = price_data.get("rsi_14", "N/A")
    atr = price_data.get("atr_14", "N/A")
    volume_ratio = price_data.get("volume_ratio", "N/A")
    high_52w = price_data.get("week_high_52", "N/A")
    low_52w = price_data.get("week_low_52", "N/A")
    # Compute pct from 52w high on the fly (not stored in DB)
    try:
        pct_from_high = round((float(close) - float(high_52w)) / float(high_52w) * 100, 1)
    except (TypeError, ValueError, ZeroDivisionError):
        pct_from_high = "N/A"

    return f"""You are a Technical Analyst for an equity research system.

Analyze the technical setup for a {direction} trade on {symbol}.

## Price Data
- Current close: ${close}
- 20-day MA: ${ma20}
- 50-day MA: ${ma50}
- RSI(14): {rsi}
- ATR(14): ${atr} (daily range proxy)
- Volume ratio vs 20d avg: {volume_ratio}x
- 52-week high: ${high_52w}  |  52-week low: ${low_52w}
- % from 52w high: {pct_from_high}%
- Market regime: {regime}

## Your Task
Write a concise technical analysis (150-200 words) covering:
1. **Trend**: Is price above/below key MAs? Momentum direction?
2. **Momentum**: RSI reading — overbought/oversold/neutral? Divergence?
3. **Volume**: Confirming or diverging from price move?
4. **Key levels**: Nearest support/resistance based on 52w range and MAs
5. **Technical verdict**: Does the chart support a {direction} thesis? Rate: SUPPORTS / NEUTRAL / CONFLICTS

Be specific. Use the numbers. No generic commentary."""


def _news_prompt(symbol: str, direction: str, news_items: list, regime: str) -> str:
    # Format top 5 news items
    if not news_items:
        news_block = "No recent news available."
    else:
        lines = []
        for i, n in enumerate(news_items[:5], 1):
            title = n.get("title", "")
            sentiment = n.get("sentiment", "")
            event_type = n.get("event_type", "")
            novelty = n.get("novelty_score", "")
            source = n.get("source", "")
            lines.append(
                f"{i}. [{event_type}] {title}\n"
                f"   Sentiment: {sentiment} | Novelty: {novelty} | Source: {source}"
            )
        news_block = "\n".join(lines)

    return f"""You are a News & Catalyst Analyst for an equity research system.

Evaluate the news catalyst quality for a {direction} trade on {symbol}.

## Recent News
{news_block}

## Market Regime: {regime}

## Your Task
Write a concise news analysis (150-200 words) covering:
1. **Catalyst type**: What is the primary driver? (earnings, product launch, macro, regulatory, etc.)
2. **Directional alignment**: Do the headlines support {direction}? Strongly / weakly / not at all?
3. **Novelty**: Is this fresh information or stale/priced-in news?
4. **Source quality**: Tier-1 outlet, company IR, social media rumor?
5. **Catalyst durability**: One-day pop or multi-day thesis?
6. **News verdict**: Rate catalyst strength: STRONG / MODERATE / WEAK / ABSENT

Be specific. Reference the actual headlines. Flag conflicts if sentiment is mixed."""


def _risk_prompt(symbol: str, direction: str, price_data: dict, score_components: dict) -> str:
    atr = price_data.get("atr_14", "N/A")
    close = price_data.get("close_price", "N/A")
    rsi = price_data.get("rsi_14", "N/A")
    risk_penalty = score_components.get("RiskPenalty", 0)

    # Compute ATR-based levels if possible
    try:
        stop_dist = float(atr) * 1.2
        target_dist = stop_dist * 2.0  # 2R
        if direction == "LONG":
            stop_price = round(float(close) - stop_dist, 2)
            target_price = round(float(close) + target_dist, 2)
        else:
            stop_price = round(float(close) + stop_dist, 2)
            target_price = round(float(close) - target_dist, 2)
        levels_block = (
            f"- Suggested stop (1.2×ATR): ${stop_price}\n"
            f"- Suggested target (2R):    ${target_price}\n"
            f"- Risk/Reward ratio: 1:2"
        )
    except (TypeError, ValueError):
        levels_block = "- Could not compute levels (ATR unavailable)"

    return f"""You are a Risk Manager for an equity research system.

Assess the risk profile for a {direction} trade on {symbol}.

## Risk Inputs
- Current close: ${close}
- ATR(14): ${atr}
- RSI(14): {rsi}
- System risk penalty score: {risk_penalty}/15 (higher = more risk flags)

## ATR-Based Trade Levels
{levels_block}

## Score Components
{json.dumps(score_components, indent=2)}

## Your Task
Write a concise risk assessment (150-200 words) covering:
1. **Primary risk**: What is the #1 thing that kills this trade?
2. **Stop logic**: Is the ATR-based stop placement reasonable here? Any gap risk?
3. **RSI risk**: Counter-trend concern? (SHORT with RSI<30 or LONG with RSI>70?)
4. **Position sizing note**: Given ATR%, should this be full/half/quarter size?
5. **Scenario if wrong**: What does the chart look like at -1R? Is there a logical exit?
6. **Risk verdict**: MANAGEABLE / ELEVATED / HIGH

Be specific. Reference actual numbers. This is used to size real paper trades."""


def _synthesis_prompt(
    symbol: str,
    direction: str,
    action_label: str,
    score_components: dict,
    technical_report: str,
    news_report: str,
    risk_report: str,
) -> str:
    total_score = sum(
        v for k, v in score_components.items() if k != "RiskPenalty"
    ) - score_components.get("RiskPenalty", 0)

    return f"""You are the Chief Investment Strategist synthesizing a team of analysts.

## Trade Setup: {direction} {symbol} — Label: {action_label}
## Composite Score: {total_score:.1f} / 85

## Score Breakdown
{json.dumps(score_components, indent=2)}

## Technical Analyst Report
{technical_report}

## News & Catalyst Report
{news_report}

## Risk Manager Report
{risk_report}

## Your Task
Synthesize the above into a final investment thesis. Output EXACTLY this JSON structure:

{{
  "conviction": "HIGH" | "MEDIUM" | "LOW",
  "summary": "<2-3 sentence thesis for the daily report. State: what the catalyst is, why the technicals confirm, and the key risk. Be specific — use symbol name, direction, and actual data points.>",
  "synthesis": "<Full 200-250 word synthesis narrative. Cover: analyst agreement/disagreement, strongest supporting factor, biggest concern, and final recommendation logic. This goes in the detailed report.>",
  "key_bull_points": ["<point 1>", "<point 2>", "<point 3>"],
  "key_bear_points": ["<point 1>", "<point 2>"]
}}

Conviction guide:
- HIGH: Technical + News both confirm direction, risk is manageable, score is strong
- MEDIUM: Mixed signals from analysts, or one layer is weak
- LOW: Significant analyst disagreement, elevated risk, or catalyst is weak/absent

Return only valid JSON. No markdown fences."""


# ─────────────────────────────────────────────
# LLM call helpers
# ─────────────────────────────────────────────

def _call_anthropic(prompt: str, model: str = "claude-haiku-4-5", max_tokens: int = 600) -> str:
    """Call Anthropic API."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def _call_groq(prompt: str, model: str = "llama-3.3-70b-versatile", max_tokens: int = 600) -> str:
    """Call Groq API."""
    import requests
    headers = {
        "Authorization": f"Bearer {os.environ.get('GROQ_API_KEY')}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _call_gemini(prompt: str, max_tokens: int = 600) -> str:
    """Call Google Gemini API."""
    import requests
    api_key = os.environ.get("GOOGLE_API_KEY")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.3},
    }
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


def _llm_call(prompt: str, provider: str, is_synthesis: bool = False) -> tuple[str, str]:
    """
    Route a prompt to the right provider.
    Returns (response_text, provider_used).
    Falls back down the chain: anthropic → groq → gemini → rule-based.
    """
    order = []
    if provider == "anthropic":
        order = ["anthropic"]
    elif provider == "groq":
        order = ["groq"]
    elif provider == "google":
        order = ["gemini"]
    elif provider == "auto":
        order = ["anthropic", "groq", "gemini"]
    else:
        order = ["groq", "gemini", "anthropic"]

    max_tokens = 1000 if is_synthesis else 600

    for p in order:
        try:
            if p == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
                model = "claude-sonnet-4-6" if is_synthesis else "claude-haiku-4-5"
                return _call_anthropic(prompt, model=model, max_tokens=max_tokens), "anthropic"
            elif p == "groq" and os.environ.get("GROQ_API_KEY"):
                return _call_groq(prompt, max_tokens=max_tokens), "groq"
            elif p == "gemini" and os.environ.get("GOOGLE_API_KEY"):
                return _call_gemini(prompt, max_tokens=max_tokens), "gemini"
        except Exception as e:
            logger.warning(f"Provider {p} failed: {e}")
            time.sleep(1)

    return "", "none"


# ─────────────────────────────────────────────
# Rule-based fallbacks (no LLM needed)
# ─────────────────────────────────────────────

def _rule_based_technical(symbol: str, direction: str, price_data: dict) -> str:
    rsi = price_data.get("rsi_14", 50)
    close = price_data.get("close_price", 0)
    ma20 = price_data.get("ma_20", 0)
    verdict = "SUPPORTS"
    notes = []
    try:
        if direction == "LONG":
            if float(close) > float(ma20):
                notes.append("Price above 20-day MA (bullish)")
            else:
                notes.append("Price below 20-day MA (bearish)")
                verdict = "CONFLICTS"
            if float(rsi) > 70:
                notes.append(f"RSI {rsi} — overbought, watch for pullback")
                verdict = "NEUTRAL"
            elif float(rsi) < 40:
                notes.append(f"RSI {rsi} — oversold, potential reversal zone")
        else:  # SHORT
            if float(close) < float(ma20):
                notes.append("Price below 20-day MA (bearish, supports short)")
            else:
                notes.append("Price above 20-day MA — short into strength, caution")
                verdict = "NEUTRAL"
            if float(rsi) < 30:
                notes.append(f"RSI {rsi} — oversold, risky short entry")
                verdict = "CONFLICTS"
    except (TypeError, ValueError):
        notes.append("Insufficient data for detailed technical analysis.")
    return (
        f"Rule-based technical summary for {direction} {symbol}:\n"
        + "\n".join(f"- {n}" for n in notes)
        + f"\n\nTechnical verdict: {verdict}"
    )


def _rule_based_news(symbol: str, direction: str, news_items: list) -> str:
    if not news_items:
        return f"No recent news found for {symbol}. Catalyst strength: ABSENT."
    sentiments = [n.get("sentiment", "neutral") for n in news_items[:5]]
    pos = sum(1 for s in sentiments if "positive" in str(s).lower() or "bullish" in str(s).lower())
    neg = sum(1 for s in sentiments if "negative" in str(s).lower() or "bearish" in str(s).lower())
    top_title = news_items[0].get("title", "N/A")
    strength = "STRONG" if pos > neg else ("WEAK" if neg > pos else "MODERATE")
    return (
        f"Rule-based news summary for {symbol} ({len(news_items)} articles):\n"
        f"- Top headline: {top_title}\n"
        f"- Positive signals: {pos} | Negative signals: {neg}\n"
        f"- Direction alignment for {direction}: {'SUPPORTS' if (direction=='LONG' and pos>neg) or (direction=='SHORT' and neg>pos) else 'CONFLICTS'}\n"
        f"\nNews verdict: {strength}"
    )


def _rule_based_risk(symbol: str, direction: str, price_data: dict, score_components: dict) -> str:
    risk_penalty = score_components.get("RiskPenalty", 0)
    rsi = price_data.get("rsi_14", 50)
    verdict = "MANAGEABLE"
    flags = []
    try:
        if risk_penalty >= 8:
            flags.append(f"High system risk penalty ({risk_penalty}/15)")
            verdict = "ELEVATED"
        if direction == "SHORT" and float(rsi) < 30:
            flags.append(f"Shorting into oversold RSI ({rsi}) — elevated squeeze risk")
            verdict = "ELEVATED"
        if direction == "LONG" and float(rsi) > 75:
            flags.append(f"Buying overbought RSI ({rsi}) — pullback risk")
        atr = price_data.get("atr_14")
        close = price_data.get("close_price")
        if atr and close:
            atr_pct = float(atr) / float(close) * 100
            if atr_pct > 4:
                flags.append(f"High ATR% ({atr_pct:.1f}%) — reduce position size")
                verdict = "ELEVATED"
    except (TypeError, ValueError):
        pass
    if not flags:
        flags.append("No major risk flags detected.")
    return (
        f"Rule-based risk summary for {direction} {symbol}:\n"
        + "\n".join(f"- {f}" for f in flags)
        + f"\n\nRisk verdict: {verdict}"
    )


def _rule_based_synthesis(
    symbol: str,
    direction: str,
    action_label: str,
    score_components: dict,
    technical_report: str,
    news_report: str,
    risk_report: str,
) -> ThesisResult:
    """Deterministic fallback when all LLMs fail."""
    risk_penalty = score_components.get("RiskPenalty", 0)
    event_edge = score_components.get("EventEdge", 0)
    total = sum(v for k, v in score_components.items() if k != "RiskPenalty") - risk_penalty

    if total >= 70 and risk_penalty < 6:
        conviction = "HIGH"
    elif total >= 55 or risk_penalty >= 8:
        conviction = "MEDIUM"
    else:
        conviction = "LOW"

    summary = (
        f"{symbol} shows a {action_label} {direction} setup with composite score {total:.0f}/85. "
        f"Event edge is {'strong' if event_edge >= 18 else 'moderate' if event_edge >= 12 else 'weak'} "
        f"and risk penalty is {'elevated' if risk_penalty >= 8 else 'acceptable'}. "
        f"Conviction: {conviction}."
    )

    return ThesisResult(
        summary=summary,
        conviction=conviction,
        technical_report=technical_report,
        news_report=news_report,
        risk_report=risk_report,
        synthesis_report=summary,
        agents_used=["rule-based"],
        provider_used="rule-based",
        fallback=True,
    )


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────

def generate_multi_agent_thesis(
    symbol: str,
    direction: str,
    score_components: dict,
    price_data: dict,
    news_items: list,
    market_regime: str,
    action_label: str,
    provider: str = "auto",
) -> ThesisResult:
    """
    Run the 4-agent pipeline and return a ThesisResult.

    Parameters
    ----------
    symbol          : e.g. "NVDA"
    direction       : "LONG" or "SHORT"
    score_components: dict with keys EventEdge, MarketConf, RegimeFit, RelOpp, RiskPenalty
    price_data      : dict from price_snapshots row (close, ma20, ma50, rsi_14, atr_14, ...)
    news_items      : list of news_articles dicts for this symbol
    market_regime   : "bull" | "bear" | "neutral" | "choppy"
    action_label    : "ACTIONABLE" | "WATCHLIST" | "MONITOR"
    provider        : "auto" | "anthropic" | "groq" | "google" | "none"
    """

    if provider == "none":
        tech = _rule_based_technical(symbol, direction, price_data)
        news = _rule_based_news(symbol, direction, news_items)
        risk = _rule_based_risk(symbol, direction, price_data, score_components)
        return _rule_based_synthesis(symbol, direction, action_label, score_components, tech, news, risk)

    # ── Agent 1: Technical ──
    tech_prompt = _technical_prompt(symbol, direction, price_data, market_regime)
    tech_text, tech_provider = _llm_call(tech_prompt, provider, is_synthesis=False)
    if not tech_text:
        tech_text = _rule_based_technical(symbol, direction, price_data)

    # ── Agent 2: News ──
    news_prompt = _news_prompt(symbol, direction, news_items, market_regime)
    news_text, news_provider = _llm_call(news_prompt, provider, is_synthesis=False)
    if not news_text:
        news_text = _rule_based_news(symbol, direction, news_items)

    # ── Agent 3: Risk ──
    risk_prompt_text = _risk_prompt(symbol, direction, price_data, score_components)
    risk_text, risk_provider = _llm_call(risk_prompt_text, provider, is_synthesis=False)
    if not risk_text:
        risk_text = _rule_based_risk(symbol, direction, price_data, score_components)

    # ── Agent 4: Synthesis ──
    synth_prompt = _synthesis_prompt(
        symbol, direction, action_label, score_components,
        tech_text, news_text, risk_text
    )
    synth_raw, synth_provider = _llm_call(synth_prompt, provider, is_synthesis=True)

    if not synth_raw:
        return _rule_based_synthesis(
            symbol, direction, action_label, score_components, tech_text, news_text, risk_text
        )

    # Parse synthesis JSON
    try:
        # Strip markdown fences if present
        clean = synth_raw.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        synth_data = json.loads(clean.strip())

        return ThesisResult(
            summary=synth_data.get("summary", ""),
            conviction=synth_data.get("conviction", "MEDIUM"),
            technical_report=tech_text,
            news_report=news_text,
            risk_report=risk_text,
            synthesis_report=synth_data.get("synthesis", synth_raw),
            agents_used=["TechnicalAgent", "NewsAgent", "RiskAgent", "SynthesisAgent"],
            provider_used=synth_provider,
            fallback=False,
        )

    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Synthesis JSON parse failed for {symbol}: {e}. Using raw text.")
        # Use raw text as summary fallback
        return ThesisResult(
            summary=synth_raw[:300],
            conviction="MEDIUM",
            technical_report=tech_text,
            news_report=news_text,
            risk_report=risk_text,
            synthesis_report=synth_raw,
            agents_used=["TechnicalAgent", "NewsAgent", "RiskAgent", "SynthesisAgent"],
            provider_used=synth_provider,
            fallback=False,
        )