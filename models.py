"""
Data models for EIA Energy Research Assistant
Author: Srikaran Anand (fsrikar@okstate.edu), Oklahoma State University
Course: Agentic AI Systems - Capstone Project (Option 4: Research Assistant)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime


@dataclass
class Conversation:
    """Represents a chat conversation session."""
    id: str
    title: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
        }


@dataclass
class Message:
    """Represents a single message in a conversation."""
    id: str
    conversation_id: str
    role: str  # "user" | "assistant"
    content: str
    metadata: Optional[Dict[str, Any]] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata or {},
            "created_at": self.created_at,
        }


@dataclass
class EiaCache:
    """Cached EIA API response with TTL."""
    id: str
    cache_key: str
    response_data: str  # JSON string
    expires_at: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "cache_key": self.cache_key,
            "response_data": self.response_data,
            "expires_at": self.expires_at,
            "created_at": self.created_at,
        }


@dataclass
class AgentLog:
    """Observability log entry for agent traces."""
    id: str
    conversation_id: Optional[str]
    event_type: str  # "guardrail", "intent", "rag_retrieve", "tool_call", "synthesis", "eval", "error"
    details: Dict[str, Any]
    duration_ms: Optional[float] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "event_type": self.event_type,
            "details": self.details,
            "duration_ms": self.duration_ms,
            "created_at": self.created_at,
        }


@dataclass
class RagChunk:
    """A knowledge base chunk for Retrieval Augmented Generation."""
    id: str
    source: str
    content: str
    embedding: Optional[str] = None  # JSON string of embedding vector
    metadata: Optional[Dict[str, Any]] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "content": self.content,
            "embedding": self.embedding,
            "metadata": self.metadata or {},
            "created_at": self.created_at,
        }


@dataclass
class AgentResult:
    """Result returned by the agent's ReAct loop."""
    answer: str
    steps: list
    evaluation: Dict[str, Any]
    data_used: Optional[Dict[str, Any]] = None
    chart_data: Optional[Dict[str, Any]] = None
    chart_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "steps": self.steps,
            "evaluation": self.evaluation,
            "data_used": self.data_used,
            "chart_data": self.chart_data,
            "chart_type": self.chart_type,
        }


@dataclass
class GuardrailResult:
    """Result from input validation guardrails."""
    passed: bool
    reason: str
    sanitized_input: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "reason": self.reason,
            "sanitized_input": self.sanitized_input,
        }


@dataclass
class EvalResult:
    """LLM-as-Judge evaluation result with 3-dimension scoring."""
    score: float           # Overall score 1-5
    reasoning: str
    factual_accuracy: float  # 1-5
    relevance: float         # 1-5
    completeness: float      # 1-5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "reasoning": self.reasoning,
            "factual_accuracy": self.factual_accuracy,
            "relevance": self.relevance,
            "completeness": self.completeness,
        }


@dataclass
class IntentResult:
    """Detected intent and entities from user query."""
    intent: str  # "analyze" | "trend" | "compare" | "forecast" | "latest" | "explain" | "explore"
    entities: Dict[str, Any]
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "entities": self.entities,
            "confidence": self.confidence,
        }
