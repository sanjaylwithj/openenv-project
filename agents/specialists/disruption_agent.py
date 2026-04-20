"""
agents/specialists/disruption_agent.py
GPT-4o-mini powered disruption risk analyst.
Assesses active disruptions and broadcasts lane risk alerts.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, Optional
from agents.base_agent import BaseAgent
from agents.communication.message_types import (
    AgentName, DisruptionReport, MessageType
)
logger = logging.getLogger("DisruptionAgent")
class DisruptionAgent(BaseAgent):
    AGENT_NAME  = "DisruptionAgent"
    MODEL       = "gpt-4o-mini"
    PROMPT_FILE = "disruption.txt"
    def assess(
        self,
        observation: dict,
        target_demand_node: Optional[str] = None,
    ) -> DisruptionReport:
        """
        Analyze active disruptions and return a structured risk report.
        Args:
            observation:         Full observation dict from GET /observation
            target_demand_node:  If set, focus analysis on this node specifically
        """
        active = observation.get("active_disruptions", [])
        lanes  = observation.get("lanes", [])
        day    = observation.get("episode_day", 0)
        user_msg = self._build_user_message(
            active, lanes, day, target_demand_node,
            observation.get("weather_forecast", {}),
        )
        logger.info(
            f"[DisruptionAgent] Assessing {len(active)} disruption(s) "
            f"on Day {day}"
            + (f" for {target_demand_node}" if target_demand_node else "")
        )
        raw = self.call(user_msg)
        return self._parse_report(raw)
    # ── Message building ──────────────────────────────────────
    def _build_user_message(
        self,
        disruptions: list,
        lanes:       list,
        day:         int,
        target_node: Optional[str],
        weather:     dict,
    ) -> str:
        # Format disruptions
        if not disruptions:
            dis_block = "NONE — all lanes are currently active."
        else:
            rows = []
            for d in disruptions:
                rows.append(
                    f"  Event:    {d.get('event_type', 'unknown').upper()}\n"
                    f"  Nodes:    {d.get('affected_nodes', [])}\n"
                    f"  Lanes:    {d.get('affected_lanes', [])}\n"
                    f"  Severity: {d.get('severity', 0):.2f}\n"
                    f"  Duration: {d.get('estimated_duration_days', '?')}d\n"
                    f"  Uncertainty: {d.get('uncertainty', 0):.2f}\n"
                    f"  Started:  Day {d.get('started_day', 0)} "
                    f"(active {day - d.get('started_day', 0)}d so far)\n"
                )
            dis_block = "\n---\n".join(rows)
        # Format lane statuses
        inactive_lanes = [
            f"  {l.get('lane_id')} ({l.get('from_node')}→{l.get('to_node')}): "
            f"congestion={l.get('congestion_level', 0):.2f}"
            for l in lanes
            if not l.get("is_active", True) or l.get("congestion_level", 0) > 0.3
        ]
        lane_block = "\n".join(inactive_lanes) if inactive_lanes else "All lanes active."
        # Weather
        weather_block = (
            "\n".join(f"  {node}: {prob:.0%} storm probability"
                      for node, prob in weather.items() if prob > 0.15)
            or "No significant weather risk."
        )
        target_line = (
            f"\nFOCUS NODE: {target_node} (analyze risk specifically for orders to this node)"
            if target_node else ""
        )
        return f"""CURRENT SIMULATION STATE — Day {day}
{target_line}
=== ACTIVE DISRUPTIONS ({len(disruptions)}) ===
{dis_block}
=== LANE STATUS (degraded lanes only) ===
{lane_block}
=== WEATHER FORECAST (3-day risk) ===
{weather_block}
Assess the overall supply chain risk and provide your structured JSON report."""
    # ── Response parsing ──────────────────────────────────────
    def _parse_report(self, raw: Dict[str, Any]) -> DisruptionReport:
        try:
            return DisruptionReport(
                risk_level=raw.get("risk_level", "low"),
                blocked_lanes=raw.get("blocked_lanes", []),
                affected_demand_nodes=raw.get("affected_demand_nodes", []),
                days_until_clear=raw.get("days_until_clear"),
                recommendation=raw.get("recommendation", "safe_to_route"),
                disruption_summary=raw.get("disruption_summary", "No disruptions."),
                severity_score=float(raw.get("severity_score", 0.0)),
            )
        except Exception as e:
            logger.error(f"[DisruptionAgent] Parse error: {e} — raw: {raw}")
            return self._parse_report(self._fallback_response())
    def _fallback_response(self) -> Dict[str, Any]:
        return {
            "risk_level":            "moderate",
            "blocked_lanes":         [],
            "affected_demand_nodes": [],
            "days_until_clear":      None,
            "recommendation":        "avoid_standard",
            "disruption_summary":    "Assessment unavailable — treating as moderate risk.",
            "severity_score":        0.5,
        }