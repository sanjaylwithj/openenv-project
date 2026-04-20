"""
agents/memory/shared_context.py
Shared episodic memory — all agents read from and write to this.
Tracks decisions, rewards, budget health, alerts, and learning signals.
"""
from __future__ import annotations
from collections import deque
from typing import Any, Deque, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
@dataclass
class DecisionRecord:
    step:              int
    day:               int
    order_id:          str
    sla_tier:          str
    demand_node:       str
    units:             int
    deadline_day:      int
    slack_days:        int
    # Agent proposals
    routing_proposal:  str
    disruption_risk:   str
    budget_approved:   bool
    budget_tier:       str
    # Final outcome
    final_decision:    str
    overrode_agent:    Optional[str]
    confidence:        float
    # Reward (filled after step)
    reward_total:      Optional[float] = None
    on_time:           Optional[bool]  = None
    cost_usd:          Optional[float] = None
    delivery_reward:   Optional[float] = None
    cost_efficiency:   Optional[float] = None
    sla_compliance:    Optional[float] = None
    timestamp:         datetime = field(default_factory=datetime.utcnow)
@dataclass
class DisruptionAlert:
    alert_id:       str
    day:            int
    risk_level:     str           # low | moderate | critical
    blocked_lanes:  List[str]
    affected_nodes: List[str]
    summary:        str
    resolved:       bool = False
class SharedContext:
    """
    Central episodic memory shared by all agents.
    Thread-safe for sequential access (standard FastAPI single-process use).
    """
    def __init__(self, task_id: str, max_history: int = 500):
        self.task_id     = task_id
        self.episode_id  = f"{task_id}_{datetime.utcnow().strftime('%H%M%S')}"
        # ── Episode state ────────────────────────────────────
        self.episode_day:       int   = 0
        self.max_days:          int   = 0
        self.budget_remaining:  float = 0.0
        self.budget_initial:    float = 0.0
        self.cumulative_cost:   float = 0.0
        self.fulfilled:         int   = 0
        self.late_deliveries:   int   = 0
        self.total_steps:       int   = 0
        self.spot_market_premium: float = 1.0
        # ── Decision history ─────────────────────────────────
        self._decisions: Deque[DecisionRecord] = deque(maxlen=max_history)
        # ── Alert history ────────────────────────────────────
        self._alerts: List[DisruptionAlert] = []
        # ── Per-agent performance tracking ───────────────────
        self.agent_stats: Dict[str, Dict[str, Any]] = {
            "RoutingAgent":    {"calls": 0, "tokens_used": 0, "avg_confidence": 0.0,
                                "proposals_accepted": 0, "proposals_overridden": 0},
            "DisruptionAgent": {"calls": 0, "tokens_used": 0, "alerts_raised": 0},
            "BudgetGuardian":  {"calls": 0, "tokens_used": 0, "approvals": 0,
                                "denials": 0, "overrides_applied": 0},
            "Orchestrator":    {"calls": 0, "tokens_used": 0, "overrides_made": 0,
                                "avg_confidence": 0.0},
        }
        # ── Running KPIs ─────────────────────────────────────
        self._reward_history: List[float] = []
    # ── Episode lifecycle ─────────────────────────────────────
    def initialize(self, obs: dict) -> None:
        """Called once at episode start with the first observation."""
        self.episode_day      = obs.get("episode_day", 0)
        self.max_days         = obs.get("max_days", 7)
        self.budget_remaining = obs.get("budget_remaining", 0.0)
        self.budget_initial   = obs.get("budget_remaining", 0.0)
        self.spot_market_premium = obs.get("spot_market_premium", 1.0)
        self._decisions.clear()
        self._alerts.clear()
        self._reward_history.clear()
        for stats in self.agent_stats.values():
            for k in stats:
                stats[k] = 0 if isinstance(stats[k], int) else 0.0
    def update_from_observation(self, obs: dict) -> None:
        """Sync running state from latest observation."""
        self.episode_day        = obs.get("episode_day", self.episode_day)
        self.budget_remaining   = obs.get("budget_remaining", self.budget_remaining)
        self.cumulative_cost    = obs.get("cumulative_cost", self.cumulative_cost)
        self.fulfilled          = obs.get("fulfilled_orders", self.fulfilled)
        self.spot_market_premium = obs.get("spot_market_premium", self.spot_market_premium)
    # ── Decision recording ────────────────────────────────────
    def record_decision(self, record: DecisionRecord) -> None:
        self._decisions.append(record)
        self.total_steps += 1
    def update_last_decision_outcome(
        self,
        reward_total:    float,
        on_time:         bool,
        cost_usd:        float,
        delivery_reward: float,
        cost_efficiency: float,
        sla_compliance:  float,
    ) -> None:
        """Fill in reward fields after the /step response comes back."""
        if not self._decisions:
            return
        last = self._decisions[-1]
        last.reward_total    = reward_total
        last.on_time         = on_time
        last.cost_usd        = cost_usd
        last.delivery_reward = delivery_reward
        last.cost_efficiency = cost_efficiency
        last.sla_compliance  = sla_compliance
        self._reward_history.append(reward_total)
    # ── Alert management ──────────────────────────────────────
    def add_alert(self, alert: DisruptionAlert) -> None:
        self._alerts.append(alert)
        self.agent_stats["DisruptionAgent"]["alerts_raised"] += 1
    def resolve_alerts(self, day: int) -> None:
        """Mark alerts from earlier days as resolved."""
        for a in self._alerts:
            if not a.resolved and a.day < day - 1:
                a.resolved = True
    def get_active_alerts(self) -> List[DisruptionAlert]:
        return [a for a in self._alerts if not a.resolved]
    # ── Agent stat tracking ───────────────────────────────────
    def record_agent_call(
        self,
        agent_name: str,
        tokens_used: int = 0,
        **kwargs,
    ) -> None:
        stats = self.agent_stats.get(agent_name, {})
        stats["calls"] = stats.get("calls", 0) + 1
        stats["tokens_used"] = stats.get("tokens_used", 0) + tokens_used
        for k, v in kwargs.items():
            if k in stats:
                if isinstance(stats[k], (int, float)) and isinstance(v, (int, float)):
                    stats[k] += v
                else:
                    stats[k] = v
    # ── Analytics & export ────────────────────────────────────
    @property
    def mean_reward(self) -> float:
        return sum(self._reward_history) / max(len(self._reward_history), 1)
    @property
    def on_time_rate(self) -> float:
        completed = [d for d in self._decisions if d.on_time is not None]
        if not completed:
            return 0.0
        return sum(1 for d in completed if d.on_time) / len(completed)
    @property
    def budget_utilization(self) -> float:
        return self.cumulative_cost / max(self.budget_initial, 1.0)
    @property
    def premium_route_rate(self) -> float:
        """Fraction of steps using premium routes (spot/express/split)."""
        premium = {"spot_market", "express_route", "split_shipment"}
        total = len(self._decisions)
        if total == 0:
            return 0.0
        return sum(1 for d in self._decisions if d.final_decision in premium) / total
    def last_n_decisions(self, n: int = 5) -> List[dict]:
        """Return last N decisions as dicts for LLM context injection."""
        recent = list(self._decisions)[-n:]
        return [
            {
                "step":           d.step,
                "day":            d.day,
                "order":          d.order_id,
                "sla":            d.sla_tier,
                "decision":       d.final_decision,
                "on_time":        d.on_time,
                "reward":         d.reward_total,
                "cost_usd":       d.cost_usd,
                "overrode_agent": d.overrode_agent,
            }
            for d in recent
        ]
    def to_summary_dict(self) -> dict:
        """Full context summary for API export and logging."""
        return {
            "episode_id":          self.episode_id,
            "task_id":             self.task_id,
            "episode_day":         self.episode_day,
            "max_days":            self.max_days,
            "total_steps":         self.total_steps,
            "budget_remaining":    round(self.budget_remaining, 2),
            "budget_initial":      self.budget_initial,
            "budget_utilization":  round(self.budget_utilization, 4),
            "cumulative_cost":     round(self.cumulative_cost, 2),
            "fulfilled":           self.fulfilled,
            "late_deliveries":     self.late_deliveries,
            "mean_reward":         round(self.mean_reward, 4),
            "on_time_rate":        round(self.on_time_rate, 4),
            "premium_route_rate":  round(self.premium_route_rate, 4),
            "active_alerts":       len(self.get_active_alerts()),
            "total_alerts":        len(self._alerts),
            "agent_stats":         self.agent_stats,
            "recent_decisions":    self.last_n_decisions(5),
        }