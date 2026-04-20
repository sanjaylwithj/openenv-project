"""
multi_agent_api.py
New FastAPI router for multi-agent endpoints.
Mount this into main.py with:  app.include_router(multi_agent_router)
"""
from __future__ import annotations
import os
import time
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()
from agents import OrchestratorAgent
from agents.communication.message_types import AgentName, MessageType
multi_agent_router = APIRouter(prefix="/multi-agent", tags=["Multi-Agent"])
# ── In-memory session registry ────────────────────────────────
# session_id → OrchestratorAgent instance
_sessions: Dict[str, OrchestratorAgent] = {}
# Background task results
_episode_results: Dict[str, Dict[str, Any]] = {}
_episode_status:  Dict[str, str] = {}   # "running" | "done" | "error"
# ── Request / Response models ─────────────────────────────────
class MultiAgentResetResponse(BaseModel):
    session_id:   str
    task_id:      str
    message:      str
    observation:  Dict[str, Any]
class MultiAgentStepRequest(BaseModel):
    session_id: str
class MultiAgentStepResponse(BaseModel):
    session_id:      str
    step:            int
    order_routed:    Optional[str]
    final_decision:  Optional[str]
    confidence:      Optional[float]
    disruption_risk: Optional[str]
    budget_tier:     Optional[str]
    budget_approved: Optional[bool]
    overrode_agent:  Optional[str]
    reward:          Optional[Dict[str, Any]]
    done:            bool
    message:         str
class EpisodeRunRequest(BaseModel):
    task_id:   str = "task_easy"
    max_steps: int = 200
class AgentStatusResponse(BaseModel):
    session_id:  str
    agents:      Dict[str, Any]
    episode_kpis: Dict[str, Any]
    active_alerts: List[Dict[str, Any]]
# ── Helpers ───────────────────────────────────────────────────
def _get_session(session_id: str) -> OrchestratorAgent:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. Call POST /multi-agent/reset first.")
    return _sessions[session_id]
def _api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY not set. Add it to your .env file."
        )
    return key
# ── Endpoints ─────────────────────────────────────────────────
@multi_agent_router.post("/reset", response_model=MultiAgentResetResponse)
def multi_agent_reset(
    task_id: str = Query("task_easy", description="task_easy | task_medium | task_hard")
):
    """
    Initialize a new multi-agent episode session.
    Returns a session_id to use in all subsequent calls.
    """
    api_key = _api_key()
    import uuid
    session_id = str(uuid.uuid4())[:12]
    orch = OrchestratorAgent(
        task_id=task_id,
        api_key=api_key,
        verbose=False,
    )
    # Reset the environment
    obs = orch._api_reset()
    orch.context.initialize(obs)
    orch._obs_cache = obs   # cache latest observation
    _sessions[session_id] = orch
    return MultiAgentResetResponse(
        session_id=session_id,
        task_id=task_id,
        message=f"Multi-agent session started. Use session_id='{session_id}' for subsequent calls.",
        observation=obs,
    )
@multi_agent_router.post("/step", response_model=MultiAgentStepResponse)
def multi_agent_step(request: MultiAgentStepRequest):
    """
    Run ONE orchestrated multi-agent step.
    The orchestrator calls all 3 specialists then submits the final action.
    """
    orch = _get_session(request.session_id)
    obs  = getattr(orch, "_obs_cache", {})
    if not obs:
        raise HTTPException(status_code=400, detail="No cached observation. Call /reset first.")
    orch.context.update_from_observation(obs)
    pending = obs.get("pending_orders", [])
    # No orders — advance day
    if not pending:
        next_obs = orch._api_advance_day(obs)
        orch._obs_cache = next_obs
        done = next_obs.get("done", False) or next_obs.get("episode_day", 0) >= next_obs.get("max_days", 99)
        return MultiAgentStepResponse(
            session_id=request.session_id,
            step=orch._step,
            order_routed=None,
            final_decision=None,
            confidence=None,
            disruption_risk=None,
            budget_tier=None,
            budget_approved=None,
            overrode_agent=None,
            reward=None,
            done=done,
            message="No pending orders — day advanced.",
        )
    order = orch._select_order(pending, obs)
    # Run specialist agents
    disruption_report = orch._run_disruption_agent(obs, order)
    route_proposal    = orch._run_routing_agent(order, obs, disruption_report)
    budget_decision   = orch._run_budget_guardian(route_proposal, order, obs)
    # Orchestrate final decision
    final = orch._orchestrate(order, obs, disruption_report, route_proposal, budget_decision)
    # Submit to environment
    step_result = orch._api_step(
        order_id=order["order_id"],
        decision=final.final_decision,
        reasoning=final.reasoning,
    )
    reward = step_result.get("reward", {})
    next_obs = step_result.get("observation", obs)
    done     = step_result.get("done", False)
    # Update context
    from agents.memory.shared_context import DecisionRecord
    day   = obs.get("episode_day", 0)
    slack = order.get("deadline_day", 0) - day
    orch.context.record_decision(DecisionRecord(
        step=orch._step,
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
    ))
    orch.context.update_last_decision_outcome(
        reward_total=reward.get("total", 0.0),
        on_time=reward.get("delivery_reward", 0.0) >= 1.0,
        cost_usd=step_result.get("info", {}).get("last_decision", {}).get("cost_usd", 0.0),
        delivery_reward=reward.get("delivery_reward", 0.0),
        cost_efficiency=reward.get("cost_efficiency", 0.0),
        sla_compliance=reward.get("sla_compliance", 0.0),
    )
    orch.context.update_from_observation(next_obs)
    orch._obs_cache = next_obs
    orch._step += 1
    return MultiAgentStepResponse(
        session_id=request.session_id,
        step=orch._step,
        order_routed=order["order_id"],
        final_decision=final.final_decision,
        confidence=final.confidence,
        disruption_risk=disruption_report.risk_level,
        budget_tier=budget_decision.budget_tier,
        budget_approved=budget_decision.approved,
        overrode_agent=final.overrode_agent,
        reward={
            "total":              reward.get("total"),
            "delivery_reward":    reward.get("delivery_reward"),
            "cost_efficiency":    reward.get("cost_efficiency"),
            "sla_compliance":     reward.get("sla_compliance"),
            "disruption_penalty": reward.get("disruption_penalty"),
            "reasoning":          reward.get("reasoning"),
        },
        done=done,
        message=f"Step {orch._step} complete — decision: {final.final_decision}",
    )
@multi_agent_router.get("/agents/status", response_model=AgentStatusResponse)
def agent_status(session_id: str = Query(..., description="Session ID from /reset")):
    """
    Get the current status and stats of all agents in this session.
    """
    orch = _get_session(session_id)
    ctx  = orch.context
    alerts = [
        {
            "alert_id":   a.alert_id,
            "day":        a.day,
            "risk_level": a.risk_level,
            "lanes":      a.blocked_lanes,
            "nodes":      a.affected_nodes,
            "summary":    a.summary,
        }
        for a in ctx.get_active_alerts()
    ]
    return AgentStatusResponse(
        session_id=session_id,
        agents={
            "Orchestrator": {
                "model":        orch.MODEL,
                **orch.tokens_used,
                "est_cost_usd": round(orch.estimated_cost_usd, 6),
            },
            "RoutingAgent": {
                "model":        orch.routing_agent.MODEL,
                **orch.routing_agent.tokens_used,
                "est_cost_usd": round(orch.routing_agent.estimated_cost_usd, 6),
                **ctx.agent_stats.get("RoutingAgent", {}),
            },
            "DisruptionAgent": {
                "model":        orch.disruption_agent.MODEL,
                **orch.disruption_agent.tokens_used,
                "est_cost_usd": round(orch.disruption_agent.estimated_cost_usd, 6),
                **ctx.agent_stats.get("DisruptionAgent", {}),
            },
            "BudgetGuardian": {
                "model":        orch.budget_guardian.MODEL,
                **orch.budget_guardian.tokens_used,
                "est_cost_usd": round(orch.budget_guardian.estimated_cost_usd, 6),
                **ctx.agent_stats.get("BudgetGuardian", {}),
            },
        },
        episode_kpis={
            "total_steps":        ctx.total_steps,
            "episode_day":        ctx.episode_day,
            "mean_reward":        round(ctx.mean_reward, 4),
            "on_time_rate":       round(ctx.on_time_rate, 4),
            "budget_utilization": round(ctx.budget_utilization, 4),
            "budget_remaining":   round(ctx.budget_remaining, 2),
            "premium_route_rate": round(ctx.premium_route_rate, 4),
        },
        active_alerts=alerts,
    )
@multi_agent_router.get("/messages")
def get_messages(
    session_id: str   = Query(...),
    msg_type:   Optional[str] = Query(None, description="Filter by message type"),
):
    """Return the full message bus log for this session."""
    orch = _get_session(session_id)
    mt = None
    if msg_type:
        try:
            mt = MessageType(msg_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid message type: {msg_type}. Valid: {[m.value for m in MessageType]}")
    return {
        "session_id": session_id,
        "stats":      orch.bus.get_stats(),
        "messages":   orch.bus.get_history(msg_type=mt),
    }
@multi_agent_router.get("/run/{task_id}")
def run_full_episode(
    task_id:   str,
    max_steps: int = Query(200, description="Maximum steps in the episode"),
):
    """
    Run a FULL multi-agent episode end-to-end and return the graded result.
    Warning: This may take 2–5 minutes for task_hard.
    """
    valid_tasks = {"task_easy", "task_medium", "task_hard"}
    if task_id not in valid_tasks:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id}")
    api_key = _api_key()
    orch = OrchestratorAgent(task_id=task_id, api_key=api_key, verbose=False)
    result = orch.run_episode(max_steps=max_steps)
    return {
        "task_id":     task_id,
        "score":       result.get("final_grade", {}).get("score", 0),
        "passed":      result.get("final_grade", {}).get("passed", False),
        "threshold":   result.get("final_grade", {}).get("threshold"),
        "subscores":   result.get("final_grade", {}).get("subscores", {}),
        "steps_taken": result.get("total_steps"),
        "mean_reward": result.get("mean_reward"),
        "on_time_rate": result.get("on_time_rate"),
        "budget_utilization": result.get("budget_utilization"),
        "premium_route_rate": result.get("premium_route_rate"),
        "token_usage": result.get("token_usage"),
        "agent_stats": result.get("agent_stats"),
        "recent_decisions": result.get("recent_decisions"),
    }
@multi_agent_router.get("/cost")
def get_cost(session_id: str = Query(...)):
    """Get token usage and estimated API cost for this session."""
    orch = _get_session(session_id)
    return {
        "session_id": session_id,
        "token_usage": {
            "Orchestrator":    {**orch.tokens_used, "model": orch.MODEL},
            "RoutingAgent":    {**orch.routing_agent.tokens_used, "model": orch.routing_agent.MODEL},
            "DisruptionAgent": {**orch.disruption_agent.tokens_used, "model": orch.disruption_agent.MODEL},
            "BudgetGuardian":  {**orch.budget_guardian.tokens_used, "model": orch.budget_guardian.MODEL},
        },
        "estimated_cost_usd": {
            "Orchestrator":    round(orch.estimated_cost_usd, 6),
            "RoutingAgent":    round(orch.routing_agent.estimated_cost_usd, 6),
            "DisruptionAgent": round(orch.disruption_agent.estimated_cost_usd, 6),
            "BudgetGuardian":  round(orch.budget_guardian.estimated_cost_usd, 6),
            "total":           round(
                orch.estimated_cost_usd
                + orch.routing_agent.estimated_cost_usd
                + orch.disruption_agent.estimated_cost_usd
                + orch.budget_guardian.estimated_cost_usd, 6
            ),
        },
    }
@multi_agent_router.delete("/session/{session_id}")
def delete_session(session_id: str):
    """Clean up a session when done."""
    if session_id in _sessions:
        del _sessions[session_id]
        return {"message": f"Session '{session_id}' deleted."}
    raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")