"""
agents/specialists/budget_guardian.py
GPT-4o-mini powered budget controller and financial risk manager.
Approves or overrides routing proposals based on financial constraints.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, Optional
from agents.base_agent import BaseAgent
from agents.communication.message_types import BudgetDecision, RouteProposal
logger = logging.getLogger("BudgetGuardian")
# Cost multipliers (mirrored from environment.py)
COST_MULTIPLIERS = {
    "standard_route":    1.0,
    "express_route":     2.8,
    "spot_market":       4.5,   # × spot_market_premium additionally
    "split_shipment":    1.6,
    "defer_24h":         0.0,
    "defer_48h":         0.0,
    "source_alternative": 1.3,
    "partial_fulfill":   0.5,
}
BUDGET_TIERS = [
    (200_000, "healthy"),
    (50_000,  "moderate"),
    (20_000,  "tight"),
    (5_000,   "critical"),
    (0,       "depleted"),
]
def _classify_budget(budget: float) -> str:
    for threshold, tier in BUDGET_TIERS:
        if budget > threshold:
            return tier
    return "depleted"
class BudgetGuardian(BaseAgent):
    AGENT_NAME  = "BudgetGuardian"
    MODEL       = "gpt-4o-mini"
    PROMPT_FILE = "budget.txt"
    def evaluate(
        self,
        proposal:        RouteProposal,
        order:           dict,
        observation:     dict,
        episode_config:  Optional[dict] = None,
    ) -> BudgetDecision:
        """
        Evaluate whether a routing proposal is financially justified.
        Args:
            proposal:       RouteProposal from RoutingAgent
            order:          The pending order
            observation:    Full observation dict
            episode_config: Task config (max_days, budget_usd) for forecasting
        """
        budget    = observation.get("budget_remaining", 0.0)
        cost_cum  = observation.get("cumulative_cost",  0.0)
        day       = observation.get("episode_day",      0)
        premium   = observation.get("spot_market_premium", 1.0)
        max_days  = observation.get("max_days", 7)
        units     = order.get("units_required", 100)
        # Estimate cost for proposed decision
        base_cost_per_unit = 15.0   # rough estimate; environment uses lane-specific values
        mult = COST_MULTIPLIERS.get(proposal.proposal, 1.0)
        if proposal.proposal == "spot_market":
            mult *= premium
        est_cost = units * base_cost_per_unit * mult
        tier = _classify_budget(budget)
        user_msg = self._build_user_message(
            proposal, order, budget, cost_cum, est_cost,
            day, max_days, tier, premium
        )
        logger.info(
            f"[BudgetGuardian] Evaluating {proposal.proposal} "
            f"| Budget=${budget:,.0f} ({tier}) "
            f"| Est.cost=${est_cost:,.0f}"
        )
        raw = self.call(user_msg)
        return self._parse_decision(raw, proposal, tier)
    # ── Message building ──────────────────────────────────────
    def _build_user_message(
        self,
        proposal:   RouteProposal,
        order:      dict,
        budget:     float,
        cost_cum:   float,
        est_cost:   float,
        day:        int,
        max_days:   int,
        tier:       str,
        premium:    float,
    ) -> str:
        remaining_days  = max_days - day
        burn_rate       = cost_cum / max(day, 1)
        projected_spend = burn_rate * remaining_days
        shortfall       = projected_spend > budget
        return f"""=== ROUTING PROPOSAL TO EVALUATE ===
Decision:          {proposal.proposal}
Confidence:        {proposal.confidence:.0%}
RoutingAgent says: {proposal.reasoning}
Alternate option:  {proposal.alternate or 'None'}
=== ORDER ===
{self.format_order(order)}
=== FINANCIAL STATE ===
Budget remaining:      ${budget:,.0f}
Budget tier:           {tier.upper()}
Cumulative cost:       ${cost_cum:,.0f}
Spot market premium:   {premium:.2f}×
=== COST ESTIMATE FOR PROPOSAL ===
Estimated route cost:  ${est_cost:,.0f}
Post-decision budget:  ${max(0, budget - est_cost):,.0f}
=== BUDGET FORECAST ===
Current day:           {day} of {max_days}
Remaining days:        {remaining_days}
Daily burn rate:       ${burn_rate:,.0f}/day
Projected future spend:${projected_spend:,.0f}
Projected shortfall:   {'⚠ YES — will run out' if shortfall else '✓ No shortfall projected'}
Approve, deny, or override the proposal based on budget health and financial risk."""
    # ── Response parsing ──────────────────────────────────────
    def _parse_decision(
        self,
        raw:       Dict[str, Any],
        proposal:  RouteProposal,
        tier:      str,
    ) -> BudgetDecision:
        try:
            return BudgetDecision(
                approved=bool(raw.get("approved", True)),
                budget_tier=raw.get("budget_tier", tier),
                override=raw.get("override"),
                override_reason=raw.get("override_reason"),
                budget_health_score=float(raw.get("budget_health_score", 0.5)),
                projected_shortfall=bool(raw.get("projected_shortfall", False)),
                alert=raw.get("alert"),
                reasoning=raw.get("reasoning", "No reasoning."),
            )
        except Exception as e:
            logger.error(f"[BudgetGuardian] Parse error: {e}")
            return BudgetDecision(
                approved=True,
                budget_tier=tier,
                override=None,
                override_reason=None,
                budget_health_score=0.5,
                projected_shortfall=False,
                alert=None,
                reasoning="Parse error — defaulting to approval.",
            )
    def _fallback_response(self) -> Dict[str, Any]:
        return {
            "approved":            True,
            "budget_tier":         "moderate",
            "override":            None,
            "override_reason":     None,
            "budget_health_score": 0.5,
            "projected_shortfall": False,
            "alert":               "BudgetGuardian API unavailable — auto-approved with caution.",
            "reasoning":           "API unavailable — applying permissive fallback approval.",
        }