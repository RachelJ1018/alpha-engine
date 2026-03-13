"""
risk_engine.py — Hedge-fund-style risk module for Alpha Engine

What this does:
  1. Defines portfolio-level risk budget
  2. Sizes positions using ATR-based stop distance
  3. Scales risk by market regime
  4. Enforces gross exposure, max positions, and sector limits
  5. Produces a position plan for candidate ideas

Designed for event-driven swing trading, not intraday execution.

Example:
    from modules.risk_engine import (
        PortfolioConfig, RiskConfig, CandidateIdea, build_position_plan
    )

    portfolio = PortfolioConfig(
        portfolio_value=100000,
        available_cash=40000,
        current_gross_exposure=0,
        open_positions=[]
    )

    risk = RiskConfig()

    idea = CandidateIdea(
        symbol="NVDA",
        direction="SHORT",
        action="WATCHLIST",
        final_score=66,
        close_price=183.14,
        atr_14=6.15,
        sector="Semis",
        strategy_bucket="event_short"
    )

    plan = build_position_plan(
        idea=idea,
        portfolio=portfolio,
        risk=risk,
        regime="bear",
    )
    print(plan.to_dict())
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════════
# Config / Data Models
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class OpenPosition:
    symbol: str
    sector: str
    direction: str
    market_value: float
    risk_dollars: float


@dataclass
class PortfolioConfig:
    portfolio_value: float
    available_cash: float
    current_gross_exposure: float = 0.0  # total market value of open positions
    open_positions: List[OpenPosition] = field(default_factory=list)


@dataclass
class RiskConfig:
    # Portfolio-level budget
    daily_risk_budget_pct: float = 1.5
    per_trade_risk_pct: float = 0.5

    # Sizing / stop model
    stop_method: str = "atr"
    atr_multiplier: float = 1.2
    fallback_stop_pct: float = 4.0

    # Portfolio constraints
    max_gross_exposure_pct: float = 60.0
    max_positions: int = 5
    max_sector_positions: int = 2
    max_single_name_exposure_pct: float = 12.0

    # Regime scaling
    bull_risk_multiplier: float = 1.0
    neutral_risk_multiplier: float = 0.8
    bear_risk_multiplier: float = 0.6
    choppy_risk_multiplier: float = 0.5
    unknown_risk_multiplier: float = 0.7

    # Score-based scaling
    actionable_multiplier: float = 1.0
    watchlist_multiplier: float = 0.75
    monitor_multiplier: float = 0.4
    ignore_multiplier: float = 0.0

    # Optional score scaling within a bucket
    score_floor: float = 40.0
    score_ceiling: float = 85.0
    score_scaling_enabled: bool = True


@dataclass
class CandidateIdea:
    symbol: str
    direction: str  # LONG / SHORT
    action: str     # ACTIONABLE / WATCHLIST / MONITOR / IGNORE
    final_score: float
    close_price: float
    atr_14: Optional[float] = None
    sector: str = "Unknown"
    strategy_bucket: str = "event_long"
    thesis: str = ""


@dataclass
class PositionPlan:
    symbol: str
    allowed: bool
    reason: str

    direction: str
    action: str
    sector: str
    strategy_bucket: str
    regime: str

    entry_price: float
    stop_price: float
    target_price: float
    stop_distance: float

    shares: int
    position_value: float
    exposure_pct: float
    risk_dollars: float
    reward_dollars: float
    rr_ratio: float

    per_trade_risk_budget: float
    remaining_daily_risk_budget: float
    remaining_gross_exposure_capacity: float
    regime_multiplier: float
    action_multiplier: float
    score_multiplier: float

    def to_dict(self) -> Dict:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════════

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_float(v: Optional[float], default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _count_sector_positions(open_positions: List[OpenPosition], sector: str) -> int:
    return sum(1 for p in open_positions if p.sector == sector)


def _remaining_daily_risk_budget(portfolio: PortfolioConfig, risk: RiskConfig) -> float:
    budget = portfolio.portfolio_value * (risk.daily_risk_budget_pct / 100.0)
    used = sum(max(0.0, p.risk_dollars) for p in portfolio.open_positions)
    return max(0.0, budget - used)


def _remaining_gross_exposure_capacity(portfolio: PortfolioConfig, risk: RiskConfig) -> float:
    max_gross = portfolio.portfolio_value * (risk.max_gross_exposure_pct / 100.0)
    return max(0.0, max_gross - portfolio.current_gross_exposure)


def _regime_multiplier(regime: str, risk: RiskConfig) -> float:
    regime = (regime or "unknown").lower()
    mapping = {
        "bull": risk.bull_risk_multiplier,
        "neutral": risk.neutral_risk_multiplier,
        "bear": risk.bear_risk_multiplier,
        "choppy": risk.choppy_risk_multiplier,
        "unknown": risk.unknown_risk_multiplier,
    }
    return mapping.get(regime, risk.unknown_risk_multiplier)


def _action_multiplier(action: str, risk: RiskConfig) -> float:
    action = (action or "IGNORE").upper()
    mapping = {
        "ACTIONABLE": risk.actionable_multiplier,
        "WATCHLIST": risk.watchlist_multiplier,
        "MONITOR": risk.monitor_multiplier,
        "IGNORE": risk.ignore_multiplier,
    }
    return mapping.get(action, 0.0)


def _score_multiplier(score: float, risk: RiskConfig) -> float:
    if not risk.score_scaling_enabled:
        return 1.0
    s = _clamp(score, risk.score_floor, risk.score_ceiling)
    # maps floor..ceiling -> 0.7..1.15
    norm = (s - risk.score_floor) / max(1e-9, (risk.score_ceiling - risk.score_floor))
    return 0.7 + 0.45 * norm


def _stop_distance(entry_price: float, atr_14: Optional[float], risk: RiskConfig) -> float:
    atr = _safe_float(atr_14, 0.0)
    if risk.stop_method == "atr" and atr > 0:
        return max(0.01, atr * risk.atr_multiplier)
    return max(0.01, entry_price * (risk.fallback_stop_pct / 100.0))


def _compute_stop_target(entry_price: float, direction: str, stop_distance: float) -> Tuple[float, float]:
    # 2R default target
    if direction.upper() == "LONG":
        stop_price = max(0.01, entry_price - stop_distance)
        target_price = entry_price + 2.0 * stop_distance
    else:
        stop_price = entry_price + stop_distance
        target_price = max(0.01, entry_price - 2.0 * stop_distance)
    return round(stop_price, 2), round(target_price, 2)


def _sector_allowed(portfolio: PortfolioConfig, risk: RiskConfig, sector: str) -> bool:
    return _count_sector_positions(portfolio.open_positions, sector) < risk.max_sector_positions


def _positions_allowed(portfolio: PortfolioConfig, risk: RiskConfig) -> bool:
    return len(portfolio.open_positions) < risk.max_positions


# ══════════════════════════════════════════════════════════════════════════
# Core risk sizing
# ══════════════════════════════════════════════════════════════════════════

def build_position_plan(
    idea: CandidateIdea,
    portfolio: PortfolioConfig,
    risk: RiskConfig,
    regime: str,
) -> PositionPlan:
    """
    Build a position plan for a single idea under portfolio constraints.
    """
    entry_price = _safe_float(idea.close_price, 0.0)
    if entry_price <= 0:
        return PositionPlan(
            symbol=idea.symbol,
            allowed=False,
            reason="Invalid entry price",
            direction=idea.direction,
            action=idea.action,
            sector=idea.sector,
            strategy_bucket=idea.strategy_bucket,
            regime=regime,
            entry_price=0.0,
            stop_price=0.0,
            target_price=0.0,
            stop_distance=0.0,
            shares=0,
            position_value=0.0,
            exposure_pct=0.0,
            risk_dollars=0.0,
            reward_dollars=0.0,
            rr_ratio=0.0,
            per_trade_risk_budget=0.0,
            remaining_daily_risk_budget=_remaining_daily_risk_budget(portfolio, risk),
            remaining_gross_exposure_capacity=_remaining_gross_exposure_capacity(portfolio, risk),
            regime_multiplier=_regime_multiplier(regime, risk),
            action_multiplier=_action_multiplier(idea.action, risk),
            score_multiplier=_score_multiplier(idea.final_score, risk),
        )

    if not _positions_allowed(portfolio, risk):
        return _blocked_plan(idea, portfolio, risk, regime, "Max positions reached")

    if not _sector_allowed(portfolio, risk, idea.sector):
        return _blocked_plan(idea, portfolio, risk, regime, "Max sector positions reached")

    regime_mult = _regime_multiplier(regime, risk)
    action_mult = _action_multiplier(idea.action, risk)
    score_mult = _score_multiplier(idea.final_score, risk)

    if action_mult <= 0:
        return _blocked_plan(idea, portfolio, risk, regime, "Action is IGNORE")

    remaining_daily_risk = _remaining_daily_risk_budget(portfolio, risk)
    if remaining_daily_risk <= 0:
        return _blocked_plan(idea, portfolio, risk, regime, "No daily risk budget left")

    base_trade_risk = portfolio.portfolio_value * (risk.per_trade_risk_pct / 100.0)
    trade_risk_budget = base_trade_risk * regime_mult * action_mult * score_mult
    trade_risk_budget = min(trade_risk_budget, remaining_daily_risk)

    stop_distance = _stop_distance(entry_price, idea.atr_14, risk)
    stop_price, target_price = _compute_stop_target(entry_price, idea.direction, stop_distance)

    # shares sized by risk
    shares_by_risk = int(trade_risk_budget // stop_distance)
    if shares_by_risk <= 0:
        return _blocked_plan(idea, portfolio, risk, regime, "Trade risk budget too small for this volatility")

    # cap by single-name exposure
    max_single_name_value = portfolio.portfolio_value * (risk.max_single_name_exposure_pct / 100.0)
    shares_by_name_cap = int(max_single_name_value // entry_price)

    # cap by available cash and gross exposure
    remaining_exposure_cap = _remaining_gross_exposure_capacity(portfolio, risk)
    shares_by_gross_cap = int(remaining_exposure_cap // entry_price)
    shares_by_cash_cap = int(portfolio.available_cash // entry_price)

    shares = min(shares_by_risk, shares_by_name_cap, shares_by_gross_cap, shares_by_cash_cap)

    if shares <= 0:
        return _blocked_plan(idea, portfolio, risk, regime, "No capacity under cash/exposure constraints")

    position_value = round(shares * entry_price, 2)
    risk_dollars = round(shares * stop_distance, 2)
    reward_dollars = round(shares * abs(target_price - entry_price), 2)
    exposure_pct = round((position_value / portfolio.portfolio_value) * 100.0, 2)
    rr_ratio = round((reward_dollars / risk_dollars), 2) if risk_dollars > 0 else 0.0

    return PositionPlan(
        symbol=idea.symbol,
        allowed=True,
        reason="OK",
        direction=idea.direction,
        action=idea.action,
        sector=idea.sector,
        strategy_bucket=idea.strategy_bucket,
        regime=regime,
        entry_price=round(entry_price, 2),
        stop_price=stop_price,
        target_price=target_price,
        stop_distance=round(stop_distance, 2),
        shares=shares,
        position_value=position_value,
        exposure_pct=exposure_pct,
        risk_dollars=risk_dollars,
        reward_dollars=reward_dollars,
        rr_ratio=rr_ratio,
        per_trade_risk_budget=round(trade_risk_budget, 2),
        remaining_daily_risk_budget=round(remaining_daily_risk, 2),
        remaining_gross_exposure_capacity=round(remaining_exposure_cap, 2),
        regime_multiplier=round(regime_mult, 2),
        action_multiplier=round(action_mult, 2),
        score_multiplier=round(score_mult, 2),
    )


def _blocked_plan(
    idea: CandidateIdea,
    portfolio: PortfolioConfig,
    risk: RiskConfig,
    regime: str,
    reason: str,
) -> PositionPlan:
    return PositionPlan(
        symbol=idea.symbol,
        allowed=False,
        reason=reason,
        direction=idea.direction,
        action=idea.action,
        sector=idea.sector,
        strategy_bucket=idea.strategy_bucket,
        regime=regime,
        entry_price=_safe_float(idea.close_price, 0.0),
        stop_price=0.0,
        target_price=0.0,
        stop_distance=0.0,
        shares=0,
        position_value=0.0,
        exposure_pct=0.0,
        risk_dollars=0.0,
        reward_dollars=0.0,
        rr_ratio=0.0,
        per_trade_risk_budget=0.0,
        remaining_daily_risk_budget=round(_remaining_daily_risk_budget(portfolio, risk), 2),
        remaining_gross_exposure_capacity=round(_remaining_gross_exposure_capacity(portfolio, risk), 2),
        regime_multiplier=round(_regime_multiplier(regime, risk), 2),
        action_multiplier=round(_action_multiplier(idea.action, risk), 2),
        score_multiplier=round(_score_multiplier(idea.final_score, risk), 2),
    )


# ══════════════════════════════════════════════════════════════════════════
# Multi-idea portfolio planning
# ══════════════════════════════════════════════════════════════════════════

def plan_candidates(
    ideas: List[CandidateIdea],
    portfolio: PortfolioConfig,
    risk: RiskConfig,
    regime: str,
) -> List[PositionPlan]:
    """
    Build plans for multiple ideas in descending score order, consuming portfolio constraints as we go.
    """
    # sort: action strength first, then score
    action_rank = {
        "ACTIONABLE": 3,
        "WATCHLIST": 2,
        "MONITOR": 1,
        "IGNORE": 0,
    }

    ordered = sorted(
        ideas,
        key=lambda x: (action_rank.get(x.action.upper(), 0), x.final_score),
        reverse=True,
    )

    working_portfolio = PortfolioConfig(
        portfolio_value=portfolio.portfolio_value,
        available_cash=portfolio.available_cash,
        current_gross_exposure=portfolio.current_gross_exposure,
        open_positions=list(portfolio.open_positions),
    )

    plans: List[PositionPlan] = []

    for idea in ordered:
        plan = build_position_plan(idea, working_portfolio, risk, regime)
        plans.append(plan)

        if plan.allowed:
            # consume constraints
            working_portfolio.available_cash -= plan.position_value
            working_portfolio.current_gross_exposure += plan.position_value
            working_portfolio.open_positions.append(
                OpenPosition(
                    symbol=idea.symbol,
                    sector=idea.sector,
                    direction=idea.direction,
                    market_value=plan.position_value,
                    risk_dollars=plan.risk_dollars,
                )
            )

    return plans


# ══════════════════════════════════════════════════════════════════════════
# Adapter for DB rows / report integration
# ══════════════════════════════════════════════════════════════════════════

def candidate_from_row(row: Dict, sector_lookup: Optional[Dict[str, str]] = None) -> CandidateIdea:
    """
    Convert a DB row / sqlite Row / dict into CandidateIdea.
    """
    sector = "Unknown"
    if sector_lookup:
        sector = sector_lookup.get(row["symbol"], "Unknown")
    if row.get("sector"):
        sector = row["sector"]

    return CandidateIdea(
        symbol=row["symbol"],
        direction=row["direction"],
        action=row["action"],
        final_score=_safe_float(row.get("final_score"), 0.0),
        close_price=_safe_float(row.get("close_price"), 0.0),
        atr_14=_safe_float(row.get("atr_14"), 0.0),
        sector=sector,
        strategy_bucket=row.get("strategy_bucket", "event_long"),
        thesis=row.get("thesis", ""),
    )


# ══════════════════════════════════════════════════════════════════════════
# Human-readable summary
# ══════════════════════════════════════════════════════════════════════════

def summarize_plan(plan: PositionPlan) -> str:
    if not plan.allowed:
        return f"{plan.symbol}: BLOCKED — {plan.reason}"

    return (
        f"{plan.symbol} {plan.direction} | {plan.action} | "
        f"{plan.shares} shares | value ${plan.position_value:,.0f} | "
        f"risk ${plan.risk_dollars:,.0f} | stop ${plan.stop_price:.2f} | "
        f"target ${plan.target_price:.2f} | R:R 1:{plan.rr_ratio:.2f}"
    )


# ══════════════════════════════════════════════════════════════════════════
# Demo
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    portfolio = PortfolioConfig(
        portfolio_value=100000,
        available_cash=40000,
        current_gross_exposure=0,
        open_positions=[],
    )

    risk = RiskConfig(
        daily_risk_budget_pct=1.5,
        per_trade_risk_pct=0.5,
        atr_multiplier=1.2,
        max_gross_exposure_pct=60,
        max_positions=5,
        max_sector_positions=2,
        max_single_name_exposure_pct=12,
    )

    ideas = [
        CandidateIdea(
            symbol="NVDA",
            direction="SHORT",
            action="WATCHLIST",
            final_score=66,
            close_price=183.14,
            atr_14=6.15,
            sector="Semis",
            strategy_bucket="event_short",
        ),
        CandidateIdea(
            symbol="AMD",
            direction="SHORT",
            action="MONITOR",
            final_score=52,
            close_price=197.74,
            atr_14=7.10,
            sector="Semis",
            strategy_bucket="event_short",
        ),
        CandidateIdea(
            symbol="AAPL",
            direction="SHORT",
            action="MONITOR",
            final_score=50,
            close_price=255.76,
            atr_14=5.12,
            sector="MegaCap Tech",
            strategy_bucket="macro_watch",
        ),
        CandidateIdea(
            symbol="PLTR",
            direction="LONG",
            action="MONITOR",
            final_score=62,
            close_price=153.50,
            atr_14=6.68,
            sector="Software",
            strategy_bucket="event_long",
        ),
    ]

    plans = plan_candidates(ideas, portfolio, risk, regime="bear")
    for p in plans:
        print(summarize_plan(p))
        print(p.to_dict())
        print("-" * 80)