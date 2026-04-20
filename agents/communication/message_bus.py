"""
agents/communication/message_bus.py
Central publish-subscribe message bus for inter-agent communication.
Supports broadcast, targeted delivery, and full message history logging.
"""
from __future__ import annotations
import uuid
import logging
from collections import defaultdict
from datetime import datetime
from typing import Callable, Dict, List, Optional
from agents.communication.message_types import (
    AgentName, BaseMessage, MessageType
)
logger = logging.getLogger("MessageBus")
class MessageBus:
    """
    Central pub/sub message router for the multi-agent system.
    Usage:
        bus = MessageBus()
        bus.subscribe(AgentName.ROUTING, MessageType.ORDER_ASSIGNED, handler_fn)
        bus.publish(message)
    """
    def __init__(self):
        # subscriber registry: {(recipient, msg_type) → [callbacks]}
        self._subscribers: Dict[tuple, List[Callable]] = defaultdict(list)
        # full audit log of every message
        self._history: List[BaseMessage] = []
        # step counter for message tagging
        self._step: int = 0
        self._day:  int = 0
    # ── Subscription management ───────────────────────────────
    def subscribe(
        self,
        agent: AgentName,
        msg_type: MessageType,
        callback: Callable[[BaseMessage], None],
    ) -> None:
        """Register a callback for a specific agent + message type."""
        key = (agent, msg_type)
        self._subscribers[key].append(callback)
        logger.debug(f"[BUS] {agent.value} subscribed to {msg_type.value}")
    def subscribe_all(
        self,
        agent: AgentName,
        callback: Callable[[BaseMessage], None],
    ) -> None:
        """Register a callback for ALL message types for this agent."""
        for mt in MessageType:
            self.subscribe(agent, mt, callback)
    # ── Publishing ────────────────────────────────────────────
    def publish(self, message: BaseMessage) -> int:
        """
        Publish a message to all relevant subscribers.
        Returns the number of handlers invoked.
        """
        message.step_number = self._step
        message.episode_day = self._day
        self._history.append(message)
        logger.info(
            f"[BUS] Day {self._day} Step {self._step} | "
            f"{message.sender.value} → "
            f"{'BROADCAST' if not message.recipient else message.recipient.value} "
            f"[{message.message_type.value}]"
        )
        handlers_called = 0
        # Deliver to explicit recipient if set
        if message.recipient:
            key = (message.recipient, message.message_type)
            for cb in self._subscribers.get(key, []):
                try:
                    cb(message)
                    handlers_called += 1
                except Exception as e:
                    logger.error(f"[BUS] Handler error for {key}: {e}")
        else:
            # Broadcast to ALL agents subscribed to this message type
            for agent in AgentName:
                key = (agent, message.message_type)
                for cb in self._subscribers.get(key, []):
                    try:
                        cb(message)
                        handlers_called += 1
                    except Exception as e:
                        logger.error(f"[BUS] Broadcast handler error for {key}: {e}")
        return handlers_called
    # ── Helpers ───────────────────────────────────────────────
    def make_message(
        self,
        sender: AgentName,
        msg_type: MessageType,
        payload: dict,
        recipient: Optional[AgentName] = None,
    ) -> BaseMessage:
        """Factory method — creates a message with auto-generated ID."""
        return BaseMessage(
            message_id=str(uuid.uuid4())[:8],
            message_type=msg_type,
            sender=sender,
            recipient=recipient,
            timestamp=datetime.utcnow(),
            episode_day=self._day,
            step_number=self._step,
            payload=payload,
        )
    def advance_step(self) -> None:
        self._step += 1
    def advance_day(self, day: int) -> None:
        self._day = day
        self._step = 0
    def reset(self) -> None:
        """Clear history and counters for a new episode."""
        self._history.clear()
        self._step = 0
        self._day = 0
        logger.info("[BUS] Reset — new episode started")
    # ── Inspection / export ───────────────────────────────────
    def get_history(self, msg_type: Optional[MessageType] = None) -> List[dict]:
        """Return message history as serializable dicts."""
        msgs = self._history
        if msg_type:
            msgs = [m for m in msgs if m.message_type == msg_type]
        return [
            {
                "id":       m.message_id,
                "type":     m.message_type.value,
                "sender":   m.sender.value,
                "recipient": m.recipient.value if m.recipient else "BROADCAST",
                "day":      m.episode_day,
                "step":     m.step_number,
                "timestamp": m.timestamp.isoformat(),
                "payload":  m.payload,
            }
            for m in msgs
        ]
    def get_stats(self) -> dict:
        """Return bus statistics."""
        by_type: Dict[str, int] = defaultdict(int)
        by_sender: Dict[str, int] = defaultdict(int)
        for m in self._history:
            by_type[m.message_type.value] += 1
            by_sender[m.sender.value] += 1
        return {
            "total_messages": len(self._history),
            "by_type":        dict(by_type),
            "by_sender":      dict(by_sender),
            "current_day":    self._day,
            "current_step":   self._step,
        }