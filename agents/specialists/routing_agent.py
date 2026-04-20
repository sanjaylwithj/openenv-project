"""
agents/specialists/routing_agent.py
GPT-4o-mini powered logistics routing specialist.
Proposes optimal routing decisions for each pending order.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, Optional
from agents.base_agent import BaseAgent
from agents.communication.message_types import DisruptionReport, RouteProposal
logger = logging.getLogger("RoutingAgent")
VALID_DECISIONS = {
    "standard_route", "express_route", "spot_market",
    "split_shipment", "defer_24h", "defer_48h",
    "source_alternative", "partial_fulfill",
}
class RoutingAgent(BaseAgent):
    AGENT_NAME  = "RoutingAgent"
    MODEL       = "gpt-4o-mini"
    PROMPT_FILE = "routing.txt"
    def decide(
        self,
        order:              dict,
        observation:        dict,
        disruption_report:  Optional[DisruptionReport] = None,
        recent_decisions:   Optional[list]             = None,
    ) -> RouteProposal:
        """
        Propose a routing decision for a single order.
        Args:
            order:             The pending order dict
            observation:       Full observation from GET /observation
            disruption_report: Report from DisruptionAgent (if available)
            recent_decisions:  Last N decisions from SharedContext for context
        """
        user_msg = self._build_user_message(
            order, observation, disruption_report, recent_decisions or []
        )
        logger.info(
            f"[RoutingAgent] Deciding for order {order.get('order_id')} "
            f"| SLA={order.get('sla_tier')} "
            f"| Slack={order.get('deadline_day', 0) - observation.get('episode_day', 0)}d"
        )
        raw = self.call(user_msg)
        return self._parse_proposal(raw, order, observation)
    # ── Message building ──────────────────────────────────────
    def _build_user_message(
        self,
        order:             dict,
        obs:               dict,
        disruption_report: Optional[DisruptionReport],
        recent_decisions:  list,
    ) -> str:
        day    = obs.get("episode_day", 0)
        slack  = order.get("deadline_day", 0) - day
        # Format disruption context
        if disruption_report:
            dis_ctx = (
                f"Risk Level:    {disruption_report.risk_level.upper()}\n"
                f"Blocked Lanes: {disruption_report.blocked_lanes}\n"
                f"Affected Nodes:{disruption_report.affected_demand_nodes}\n"
                f"Recommendation:{disruption_report.recommendation}\n"
                f"Summary:       {disruption_report.disruption_summary}\n"
                f"Days to clear: {disruption_report.days_until_clear or 'unknown'}"
            )
        else:
            raw_dis = obs.get("active_disruptions", [])
            dis_ctx = self.format_disruptions(raw_dis) if raw_dis else "No active disruptions."
        # Format recent decision context for learning
        if recent_decisions:
            hist_lines = [
                f"  [{d.get('day', 0)}d] {d.get('decision')} → "
                f"{'✓' if d.get('on_time') else '✗'} "
                f"reward={d.get('reward', 0):.3f} "
                f"cost=${d.get('cost_usd', 0):,.0f}"
                for d in recent_decisions[-3:]
            ]
            history_ctx = "\n".join(hist_lines)
        else:
            history_ctx = "No prior decisions in this episode."
        return f"""=== ORDER TO ROUTE ===
{self.format_order(order)}
Deadline Slack: {slack} days (current Day {day}, deadline Day {order.get('deadline_day')})
=== NETWORK & BUDGET STATE ===
{self.format_budget(obs)}
=== DISRUPTION INTELLIGENCE (from DisruptionAgent) ===
{dis_ctx}
=== RECENT DECISION HISTORY (learn from this) ===
{history_ctx}
Based on this full context, propose the optimal routing decision for this order.
Consider the deadline slack, SLA tier, disruption risk, and budget state."""
    # ── Response parsing ──────────────────────────────────────
    def _parse_proposal(
        self, raw: Dict[str, Any], order: dict, obs: dict
    ) -> RouteProposal:
        proposal = raw.get("proposal", "standard_route")
        # Validate decision is legal
        if proposal not in VALID_DECISIONS:
            logger.warning(f"[RoutingAgent] Invalid proposal '{proposal}' — defaulting to standard_route")
            proposal = "standard_route"
        try:
            return RouteProposal(
                proposal=proposal,
                confidence=float(raw.get("confidence", 0.5)),
                slack_days=int(raw.get("slack_days",
                               order.get("deadline_day", 0) - obs.get("episode_day", 0))),
                reasoning=raw.get("reasoning", "No reasoning provided."),
                alternate=raw.get("alternate"),
            )
        except Exception as e:
            logger.error(f"[RoutingAgent] Parse error: {e}")
            return RouteProposal(
                proposal="standard_route",
                confidence=0.3,
                slack_days=0,
                reasoning="Parse error — defaulting to standard route.",
                alternate=None,
            )
    def _fallback_response(self) -> Dict[str, Any]:
        return {
            "proposal":   "standard_route",
            "confidence": 0.3,
            "slack_days": 3,
            "reasoning":  "API unavailable — applying conservative standard route fallback.",
            "alternate":  "express_route",
        }