"""
agents/orchestrator.py
GPT-4o powered master coordinator.
Synthesizes reports from all 3 specialist agents and makes the final routing decision.
"""
from __future__ import annotations
import json
import logging
import time
from typing import Any, Dict, List, Optional
import requests
from agents.base_agent import BaseAgent
from agents.specialists.disruption_agent import DisruptionAgent
from agents.specialists.routing_agent import RoutingAgent
from agents.specialists.budget_guardian import BudgetGuardian
from agents.communication.message_bus import MessageBus
from agents.communication.message_types import (
    AgentName, BudgetDecision, DisruptionReport,
    MessageType, OrchestratorDecision, RouteProposal,
    StepOutcome,
)
from agents.memory.shared_context import DecisionRecord, DisruptionAlert, SharedContext
logger = logging.getLogger("Orchestrator")
BASE_URL = "http://localhost:7860"
class OrchestratorAgent(BaseAgent):
    """
    Master coordinator for the multi-agent supply chain system.
    Uses GPT-4o to synthesize specialist reports and submit final decisions.
    """
    AGENT_NAME  = "Orchestrator"
    MODEL       = "gpt-4o"
    PROMPT_FILE = "orchestrator.txt"
    def __init__(
        self,
        task_id:     str = "task_easy",
        api_key:     Optional[str] = None,
        base_url:    str = BASE_URL,
        verbose:     bool = True,
    ):
        super().__init__(api_key=api_key)
        self.task_id  = task_id
        self.base_url = base_url.rstrip("/")
        self.verbose  = verbose
        # Specialist agents
        self.disruption_agent = DisruptionAgent(api_key=api_key)
        self.routing_agent    = RoutingAgent(api_key=api_key)
        self.budget_guardian  = BudgetGuardian(api_key=api_key)
        # Communication & memory
        self.bus     = MessageBus()
        self.context = SharedContext(task_id=task_id)
        self._step = 0
        self._obs_cache: Dict[str, Any] = {}  # Cache latest observation
    # ── Episode lifecycle ─────────────────────────────────────
    def run_episode(self, max_steps: int = 200) -> Dict[str, Any]:
        """
        Run a full episode from reset to terminal state.
        Returns the final grader result + episode summary.
        """
        logger.info(f"[Orchestrator] Starting episode — task={self.task_id}")
        self.bus.reset()
        # Reset environment
        obs = self._api_reset()
        self.context.initialize(obs)
        self.bus.advance_day(0)
        self._step = 0
        episode_log: List[dict] = []
        while self._step < max_steps:
            self.context.update_from_observation(obs)
            self.bus.advance_day(obs.get("episode_day", 0))
            pending = obs.get("pending_orders", [])
            if not pending:
                # No orders — advance the environment day
                obs = self._api_advance_day(obs)
                if obs.get("done", False) or obs.get("episode_day", 0) >= obs.get("max_days", 99):
                    break
                continue
            # ── Select highest-priority order ─────────────────
            order = self._select_order(pending, obs)
            if self.verbose:
                self._log_order(order, obs)
            # ── Run specialist agents in parallel (sequential here) ──
            disruption_report = self._run_disruption_agent(obs, order)
            route_proposal    = self._run_routing_agent(order, obs, disruption_report)
            budget_decision   = self._run_budget_guardian(route_proposal, order, obs)
            # ── Orchestrate final decision ────────────────────
            final = self._orchestrate(
                order, obs, disruption_report, route_proposal, budget_decision
            )
            # ── Record to shared memory ───────────────────────
            day   = obs.get("episode_day", 0)
            slack = order.get("deadline_day", 0) - day
            record = DecisionRecord(
                step=self._step,
                day=day,
                order_id=order["order_id"],
                sla_tier=order["sla_tier"],
                demand_node=order["demand_node"],
                units=order["units_required"],
                deadline_day=order["deadline_day"],
                slack_days=slack,
                routing_proposal=route_proposal.proposal,
                disruption_risk=disruption_report.risk_level,
                budget_approved=budget_decision.approved,
                budget_tier=budget_decision.budget_tier,
                final_decision=final.final_decision,
                overrode_agent=final.overrode_agent,
                confidence=final.confidence,
            )
            self.context.record_decision(record)
            # ── Submit action to environment ──────────────────
            step_result = self._api_step(
                order_id=order["order_id"],
                decision=final.final_decision,
                reasoning=final.reasoning,
            )
            # ── Update memory with reward ─────────────────────
            reward = step_result.get("reward", {})
            self.context.update_last_decision_outcome(
                reward_total=reward.get("total", 0.0),
                on_time=self._infer_on_time(reward),
                cost_usd=step_result.get("info", {}).get("last_decision", {}).get("cost_usd", 0.0),
                delivery_reward=reward.get("delivery_reward", 0.0),
                cost_efficiency=reward.get("cost_efficiency", 0.0),
                sla_compliance=reward.get("sla_compliance", 0.0),
            )
            # ── Publish STEP_RESULT to all agents ─────────────
            self.bus.publish(self.bus.make_message(
                sender=AgentName.ORCHESTRATOR,
                msg_type=MessageType.STEP_RESULT,
                payload={
                    "step":     self._step,
                    "order_id": order["order_id"],
                    "decision": final.final_decision,
                    "reward":   reward,
                },
            ))
            # ── Log step ─────────────────────────────────────
            episode_log.append({
                "step":              self._step,
                "day":               day,
                "order_id":          order["order_id"],
                "sla_tier":          order["sla_tier"],
                "decision":          final.final_decision,
                "confidence":        final.confidence,
                "disruption_risk":   disruption_report.risk_level,
                "budget_tier":       budget_decision.budget_tier,
                "budget_approved":   budget_decision.approved,
                "overrode_agent":    final.overrode_agent,
                "reward_total":      reward.get("total"),
                "expected_outcome":  final.expected_outcome,
            })
            if self.verbose:
                self._log_step_result(final, reward, step_result.get("info", {}))
            obs   = step_result.get("observation", obs)
            done  = step_result.get("done", False)
            self._step += 1
            if done:
                logger.info(f"[Orchestrator] Episode done at step {self._step}")
                break
        # ── Final grade ───────────────────────────────────────
        grade = self._api_grade()
        summary = self.context.to_summary_dict()
        summary["episode_log"]   = episode_log
        summary["final_grade"]   = grade
        summary["token_usage"]   = self._all_token_stats()
        logger.info(
            f"[Orchestrator] Episode complete — "
            f"score={grade.get('score', 0):.4f} | "
            f"steps={self._step} | "
            f"mean_reward={self.context.mean_reward:.4f}"
        )
        return summary
    # ── Agent orchestration ───────────────────────────────────
    def _run_disruption_agent(self, obs: dict, order: dict) -> DisruptionReport:
        start = time.perf_counter()
        report = self.disruption_agent.assess(obs, target_demand_node=order.get("demand_node"))
        elapsed = time.perf_counter() - start
        self.context.record_agent_call(
            "DisruptionAgent",
            tokens_used=self.disruption_agent.tokens_used["total"],
        )
        # Raise alert if critical
        if report.risk_level == "critical":
            alert = DisruptionAlert(
                alert_id=f"ALT-{self._step:03d}",
                day=obs.get("episode_day", 0),
                risk_level=report.risk_level,
                blocked_lanes=report.blocked_lanes,
                affected_nodes=report.affected_demand_nodes,
                summary=report.disruption_summary,
            )
            self.context.add_alert(alert)
            self.bus.publish(self.bus.make_message(
                sender=AgentName.DISRUPTION,
                msg_type=MessageType.LANE_RISK_ALERT,
                payload=report.model_dump(),
            ))
        logger.debug(f"[DisruptionAgent] {report.risk_level} risk | {elapsed:.2f}s")
        return report
    def _run_routing_agent(
        self,
        order:             dict,
        obs:               dict,
        disruption_report: DisruptionReport,
    ) -> RouteProposal:
        start    = time.perf_counter()
        recent   = self.context.last_n_decisions(5)
        proposal = self.routing_agent.decide(order, obs, disruption_report, recent)
        elapsed  = time.perf_counter() - start
        self.context.record_agent_call(
            "RoutingAgent",
            tokens_used=self.routing_agent.tokens_used["total"],
        )
        self.bus.publish(self.bus.make_message(
            sender=AgentName.ROUTING,
            msg_type=MessageType.ROUTE_PROPOSAL,
            recipient=AgentName.ORCHESTRATOR,
            payload=proposal.model_dump(),
        ))
        logger.debug(f"[RoutingAgent] Proposed {proposal.proposal} (conf={proposal.confidence:.0%}) | {elapsed:.2f}s")
        return proposal
    def _run_budget_guardian(
        self,
        proposal: RouteProposal,
        order:    dict,
        obs:      dict,
    ) -> BudgetDecision:
        start    = time.perf_counter()
        decision = self.budget_guardian.evaluate(proposal, order, obs)
        elapsed  = time.perf_counter() - start
        self.context.record_agent_call(
            "BudgetGuardian",
            tokens_used=self.budget_guardian.tokens_used["total"],
        )
        msg_type = MessageType.BUDGET_APPROVED if decision.approved else MessageType.BUDGET_DENIED
        self.bus.publish(self.bus.make_message(
            sender=AgentName.BUDGET,
            msg_type=msg_type,
            recipient=AgentName.ORCHESTRATOR,
            payload=decision.model_dump(),
        ))
        logger.debug(
            f"[BudgetGuardian] {'APPROVED' if decision.approved else 'DENIED'} "
            f"({decision.budget_tier}) | {elapsed:.2f}s"
        )
        return decision
    def _orchestrate(
        self,
        order:             dict,
        obs:               dict,
        disruption_report: DisruptionReport,
        route_proposal:    RouteProposal,
        budget_decision:   BudgetDecision,
    ) -> OrchestratorDecision:
        """Call GPT-4o to make the final authoritative decision."""
        user_msg = self._build_orchestration_message(
            order, obs, disruption_report, route_proposal, budget_decision
        )
        raw = self.call(user_msg)
        decision = self._parse_orchestrator_decision(raw)
        # Attach sub-agent reports for memory
        decision.disruption_report = disruption_report
        decision.route_proposal    = route_proposal
        decision.budget_decision   = budget_decision
        self.context.record_agent_call(
            "Orchestrator",
            tokens_used=self.tokens_used["total"],
        )
        self.bus.publish(self.bus.make_message(
            sender=AgentName.ORCHESTRATOR,
            msg_type=MessageType.FINAL_DECISION,
            payload=decision.model_dump(exclude={"disruption_report", "route_proposal", "budget_decision"}),
        ))
        return decision
    # ── Message building ──────────────────────────────────────
    def _build_orchestration_message(
        self,
        order:          dict,
        obs:            dict,
        dis_report:     DisruptionReport,
        route_proposal: RouteProposal,
        budget_dec:     BudgetDecision,
    ) -> str:
        day   = obs.get("episode_day", 0)
        slack = order.get("deadline_day", 0) - day
        return f"""=== ORDER TO ROUTE ===
{self.format_order(order)}
Deadline Slack: {slack}d (Day {day} of {obs.get('max_days', '?')})
=== AGENT REPORTS ===
[DisruptionAgent — gpt-4o-mini]
  Risk Level:    {dis_report.risk_level.upper()}
  Blocked Lanes: {dis_report.blocked_lanes}
  Recommendation:{dis_report.recommendation}
  Summary:       {dis_report.disruption_summary}
  Severity:      {dis_report.severity_score:.2f}
[RoutingAgent — gpt-4o-mini]
  Proposal:   {route_proposal.proposal}
  Confidence: {route_proposal.confidence:.0%}
  Slack Days: {route_proposal.slack_days}
  Reasoning:  {route_proposal.reasoning}
  Alternate:  {route_proposal.alternate or 'None'}
[BudgetGuardian — gpt-4o-mini]
  Approved:     {budget_dec.approved}
  Budget Tier:  {budget_dec.budget_tier.upper()}
  Override:     {budget_dec.override or 'None'}
  Health Score: {budget_dec.budget_health_score:.2f}
  Alert:        {budget_dec.alert or 'None'}
  Reasoning:    {budget_dec.reasoning}
=== CURRENT EPISODE KPIs ===
{self.format_budget(obs)}
On-time rate:    {obs.get('on_time_delivery_rate', 0):.1%}
Service level:   {obs.get('service_level', 0):.1%}
=== CONFLICT STATUS ===
{'⚠ CONFLICT: BudgetGuardian DENIED RoutingAgent proposal — override in effect' if not budget_dec.approved else '✓ Agents in agreement — no conflict'}
Make the FINAL routing decision, incorporating all agent input and resolving any conflicts."""
    # ── Response parsing ──────────────────────────────────────
    def _parse_orchestrator_decision(self, raw: Dict[str, Any]) -> OrchestratorDecision:
        decision = raw.get("final_decision", "standard_route")
        valid = {
            "standard_route", "express_route", "spot_market",
            "split_shipment", "defer_24h", "defer_48h",
            "source_alternative", "partial_fulfill",
        }
        if decision not in valid:
            logger.warning(f"[Orchestrator] Invalid decision '{decision}' — defaulting")
            decision = "standard_route"
        return OrchestratorDecision(
            final_decision=decision,
            confidence=float(raw.get("confidence", 0.5)),
            reasoning=raw.get("reasoning", "No reasoning."),
            overrode_agent=raw.get("overrode_agent"),
            override_justification=raw.get("override_justification"),
            risk_acknowledged=raw.get("risk_acknowledged", "low"),
            expected_outcome=raw.get("expected_outcome", "likely_on_time"),
        )
    def _fallback_response(self) -> Dict[str, Any]:
        return {
            "final_decision":        "standard_route",
            "confidence":            0.3,
            "reasoning":             "Orchestrator API unavailable — applying conservative fallback.",
            "overrode_agent":        None,
            "override_justification": None,
            "risk_acknowledged":     "moderate",
            "expected_outcome":      "at_risk",
        }
    # ── Order selection ───────────────────────────────────────
    def _select_order(self, orders: list, obs: dict) -> dict:
        """Select highest-priority order: critical SLA first, then tight deadlines."""
        day = obs.get("episode_day", 0)
        priority = {"critical": 0, "standard": 1, "flexible": 2}
        return min(
            orders,
            key=lambda o: (
                priority.get(o.get("sla_tier", "flexible"), 2),
                o.get("deadline_day", 99) - day,
            ),
        )
    # ── API helpers ───────────────────────────────────────────
    def _api_reset(self) -> dict:
        resp = requests.post(f"{self.base_url}/reset?task_id={self.task_id}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    def _api_step(self, order_id: str, decision: str, reasoning: str) -> dict:
        payload = {
            "task_id": self.task_id,
            "action": {
                "order_id":         order_id,
                "routing_decision": decision,
                "alternate_supplier": None,
                "reasoning":        reasoning,
            },
        }
        resp = requests.post(f"{self.base_url}/step", json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json()
    def _api_advance_day(self, current_obs: dict) -> dict:
        """When no pending orders — read next observation (env advances on empty queue)."""
        resp = requests.get(f"{self.base_url}/observation?task_id={self.task_id}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    def _api_grade(self) -> dict:
        resp = requests.get(f"{self.base_url}/grader/{self.task_id}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    # ── Token accounting ──────────────────────────────────────
    def _all_token_stats(self) -> dict:
        return {
            "Orchestrator":    {**self.tokens_used, "estimated_cost_usd": round(self.estimated_cost_usd, 6)},
            "RoutingAgent":    {**self.routing_agent.tokens_used,
                                "estimated_cost_usd": round(self.routing_agent.estimated_cost_usd, 6)},
            "DisruptionAgent": {**self.disruption_agent.tokens_used,
                                "estimated_cost_usd": round(self.disruption_agent.estimated_cost_usd, 6)},
            "BudgetGuardian":  {**self.budget_guardian.tokens_used,
                                "estimated_cost_usd": round(self.budget_guardian.estimated_cost_usd, 6)},
            "total_cost_usd":  round(
                self.estimated_cost_usd
                + self.routing_agent.estimated_cost_usd
                + self.disruption_agent.estimated_cost_usd
                + self.budget_guardian.estimated_cost_usd, 6
            ),
        }
    # ── Utilities ─────────────────────────────────────────────
    @staticmethod
    def _infer_on_time(reward: dict) -> bool:
        return reward.get("delivery_reward", 0.0) >= 1.0
    def _log_order(self, order: dict, obs: dict) -> None:
        day   = obs.get("episode_day", 0)
        slack = order.get("deadline_day", 0) - day
        sla = order.get("sla_tier", "unknown")
        sla_str = sla.upper() if sla else "UNKNOWN"
        print(
            f"\n{'─'*60}\n"
            f"  Step {self._step} | Day {day} | {order.get('order_id')}\n"
            f"  SLA: {sla_str:<10} "
            f"Slack: {slack}d  Units: {order.get('units_required')}\n"
            f"  Dest: {order.get('demand_node'):<15} "
            f"Budget: ${obs.get('budget_remaining', 0):,.0f}\n"
        )
    def _log_step_result(self, decision: OrchestratorDecision, reward: dict, info: dict) -> None:
        override_line = (
            f"  ⚡ Override: {decision.overrode_agent} → {decision.override_justification}"
            if decision.overrode_agent else "  ✓ Agents aligned"
        )
        print(
            f"  → {decision.final_decision.upper()} (conf={decision.confidence:.0%})\n"
            f"{override_line}\n"
            f"  Reward: {reward.get('total', 0):+.4f}  "
            f"[delivery={reward.get('delivery_reward', 0):+.2f} | "
            f"cost={reward.get('cost_efficiency', 0):+.2f} | "
            f"sla={reward.get('sla_compliance', 0):+.2f}]\n"
            f"  Budget left: ${info.get('budget_remaining', 0):,.0f}\n"
        )