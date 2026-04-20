"""
agents/communication/message_types.py
Typed Pydantic message schemas for inter-agent communication.
"""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
class MessageType(str, Enum):
    # Orchestrator → Specialists
    ORDER_ASSIGNED       = "ORDER_ASSIGNED"
    EPISODE_STARTED      = "EPISODE_STARTED"
    EPISODE_ENDED        = "EPISODE_ENDED"
    # DisruptionAgent → All
    LANE_RISK_ALERT      = "LANE_RISK_ALERT"
    DISRUPTION_CLEARED   = "DISRUPTION_CLEARED"
    # RoutingAgent → Orchestrator
    ROUTE_PROPOSAL       = "ROUTE_PROPOSAL"
    # BudgetGuardian → Orchestrator
    BUDGET_APPROVED      = "BUDGET_APPROVED"
    BUDGET_DENIED        = "BUDGET_DENIED"
    BUDGET_ALERT         = "BUDGET_ALERT"
    # Orchestrator → Environment
    FINAL_DECISION       = "FINAL_DECISION"
    # Environment → All (after step)
    STEP_RESULT          = "STEP_RESULT"
class AgentName(str, Enum):
    ORCHESTRATOR    = "Orchestrator"
    ROUTING         = "RoutingAgent"
    DISRUPTION      = "DisruptionAgent"
    BUDGET          = "BudgetGuardian"
    ENVIRONMENT     = "Environment"
class BaseMessage(BaseModel):
    message_id:  str
    message_type: MessageType
    sender:      AgentName
    recipient:   Optional[AgentName] = None   # None = broadcast
    timestamp:   datetime = Field(default_factory=datetime.utcnow)
    episode_day: int = 0
    step_number: int = 0
    payload:     Dict[str, Any] = {}
# ── Specialist report messages ────────────────────────────────
class DisruptionReport(BaseModel):
    """Output of DisruptionAgent.assess()"""
    risk_level:            str                   # low | moderate | critical
    blocked_lanes:         List[str]
    affected_demand_nodes: List[str]
    days_until_clear:      Optional[int]
    recommendation:        str
    disruption_summary:    str
    severity_score:        float
class RouteProposal(BaseModel):
    """Output of RoutingAgent.decide()"""
    proposal:   str                              # routing decision name
    confidence: float
    slack_days: int
    reasoning:  str
    alternate:  Optional[str] = None
class BudgetDecision(BaseModel):
    """Output of BudgetGuardian.evaluate()"""
    approved:              bool
    budget_tier:           str                   # healthy|moderate|tight|critical|depleted
    override:              Optional[str]         # alternative decision if denied
    override_reason:       Optional[str]
    budget_health_score:   float
    projected_shortfall:   bool
    alert:                 Optional[str]
    reasoning:             str
class OrchestratorDecision(BaseModel):
    """Final output of Orchestrator.decide()"""
    final_decision:       str
    confidence:           float
    reasoning:            str
    overrode_agent:       Optional[str]
    override_justification: Optional[str]
    risk_acknowledged:    str
    expected_outcome:     str
    # Full agent context (stored in shared memory)
    disruption_report:    Optional[DisruptionReport] = None
    route_proposal:       Optional[RouteProposal]    = None
    budget_decision:      Optional[BudgetDecision]   = None
class StepOutcome(BaseModel):
    """Result after POST /step — fed back to all agents"""
    order_id:             str
    routing_decision:     str
    reward_total:         float
    delivery_reward:      float
    cost_efficiency:      float
    sla_compliance:       float
    disruption_penalty:   float
    reward_reasoning:     str
    on_time:              bool
    cost_usd:             float
    budget_remaining:     float
    episode_day:          int