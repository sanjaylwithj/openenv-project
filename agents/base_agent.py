"""
agents/base_agent.py
Abstract base class for all OpenAI-powered agents.
Handles: client setup, retry logic, token tracking, JSON parsing, error handling.
"""
from __future__ import annotations
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, TypeVar
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError
from pydantic import BaseModel
logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
class BaseAgent(ABC):
    """
    Abstract base for all supply chain agents.
    Subclasses implement `_build_user_message()` and define their Pydantic output schema.
    """
    # Override in subclasses
    AGENT_NAME: str = "BaseAgent"
    MODEL:      str = "gpt-4o-mini"
    PROMPT_FILE: str = ""                  # filename under agents/prompts/
    def __init__(
        self,
        api_key:     Optional[str] = None,
        temperature: float = 0.1,
        max_retries: int   = 3,
        timeout:     float = 30.0,
    ):
        self._client = OpenAI(
            api_key=api_key or os.environ["OPENAI_API_KEY"],
            timeout=timeout,
        )
        self._temperature  = temperature
        self._max_retries  = max_retries
        # Token accounting
        self._total_prompt_tokens:     int = 0
        self._total_completion_tokens: int = 0
        self._total_calls:             int = 0
        # Load system prompt from file
        prompt_path = os.path.join(
            os.path.dirname(__file__), "prompts", self.PROMPT_FILE
        )
        with open(prompt_path, "r", encoding="utf-8") as f:
            self._system_prompt = f.read().strip()
        logger.info(f"[{self.AGENT_NAME}] Initialized — model={self.MODEL}")
    # ── Public interface ──────────────────────────────────────
    def call(
        self,
        user_message: str,
        output_schema: Optional[Type[T]] = None,
    ) -> Dict[str, Any]:
        """
        Make an OpenAI API call with retry logic.
        Returns parsed JSON dict. Validates against output_schema if provided.
        """
        last_error: Optional[Exception] = None
        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.MODEL,
                    temperature=self._temperature,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": self._system_prompt},
                        {"role": "user",   "content": user_message},
                    ],
                )
                # Track usage
                usage = response.usage
                if usage:
                    self._total_prompt_tokens     += usage.prompt_tokens
                    self._total_completion_tokens += usage.completion_tokens
                self._total_calls += 1
                raw = response.choices[0].message.content
                if raw is None:
                    logger.error(f"[{self.AGENT_NAME}] API returned None content")
                    raise ValueError("API response content is None")
                result = json.loads(raw)
                # Validate with Pydantic schema if provided
                if output_schema:
                    validated = output_schema(**result)
                    return validated.model_dump()
                return result
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                wait = 2 ** attempt
                logger.warning(
                    f"[{self.AGENT_NAME}] API error on attempt {attempt}/{self._max_retries}: "
                    f"{type(e).__name__} — retrying in {wait}s"
                )
                last_error = e
                time.sleep(wait)
            except json.JSONDecodeError as e:
                logger.error(f"[{self.AGENT_NAME}] JSON parse error: {e}")
                last_error = e
                break
            except Exception as e:
                logger.error(f"[{self.AGENT_NAME}] Unexpected error: {e}")
                last_error = e
                break
        # All retries exhausted — return safe fallback
        logger.error(
            f"[{self.AGENT_NAME}] All {self._max_retries} retries failed. "
            f"Last error: {last_error}. Using fallback."
        )
        return self._fallback_response()
    @abstractmethod
    def _fallback_response(self) -> Dict[str, Any]:
        """
        Return a safe default response when all API calls fail.
        Must match the agent's expected output schema.
        """
    # ── Token accounting ──────────────────────────────────────
    @property
    def tokens_used(self) -> Dict[str, int]:
        return {
            "prompt":     self._total_prompt_tokens,
            "completion": self._total_completion_tokens,
            "total":      self._total_prompt_tokens + self._total_completion_tokens,
            "calls":      self._total_calls,
        }
    @property
    def estimated_cost_usd(self) -> float:
        """Rough cost estimate based on current OpenAI pricing."""
        # gpt-4o-mini: $0.15/1M input, $0.60/1M output
        # gpt-4o:      $2.50/1M input, $10.00/1M output
        if "mini" in self.MODEL:
            return (
                self._total_prompt_tokens     * 0.00000015
                + self._total_completion_tokens * 0.00000060
            )
        else:
            return (
                self._total_prompt_tokens     * 0.0000025
                + self._total_completion_tokens * 0.000010
            )
    def reset_token_count(self) -> None:
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_calls = 0
    # ── Utility helpers for subclasses ────────────────────────
    @staticmethod
    def format_order(order: dict) -> str:
        sla_tier = order.get('sla_tier', 'unknown')
        sla_tier_str = sla_tier.upper() if sla_tier else 'UNKNOWN'
        return (
            f"Order ID:     {order.get('order_id')}\n"
            f"SKU:          {order.get('sku')}\n"
            f"Units:        {order.get('units_required')}\n"
            f"Destination:  {order.get('demand_node')}\n"
            f"SLA Tier:     {sla_tier_str}\n"
            f"Deadline:     Day {order.get('deadline_day')}\n"
            f"Value (USD):  ${order.get('value_usd', 0):,.0f}\n"
            f"Penalty/day:  ${order.get('penalty_per_day_late', 0):,.0f}"
        )
    @staticmethod
    def format_disruptions(disruptions: list) -> str:
        if not disruptions:
            return "None — all lanes active."
        lines = []
        for d in disruptions:
            lines.append(
                f"  [{d.get('event_type', 'unknown').upper()}] "
                f"Severity={d.get('severity', 0):.2f} | "
                f"Lanes={d.get('affected_lanes', [])} | "
                f"Est. duration={d.get('estimated_duration_days', '?')}d | "
                f"Uncertainty={d.get('uncertainty', 0):.2f}"
            )
        return "\n".join(lines)
    @staticmethod
    def format_budget(obs: dict) -> str:
        return (
            f"Budget remaining:   ${obs.get('budget_remaining', 0):,.0f}\n"
            f"Cumulative cost:    ${obs.get('cumulative_cost', 0):,.0f}\n"
            f"Spot market premium: {obs.get('spot_market_premium', 1.0):.2f}×\n"
            f"On-time rate so far: {obs.get('on_time_delivery_rate', 0):.1%}"
        )