from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from ..schemas.llm import ChatMessage


class ConversationTurn(BaseModel):
    """A single turn in the conversation."""
    user_input: str
    assistant_response: str
    intent: Optional[str] = None
    timestamp: datetime
    quality_score: Optional[float] = None


class ConversationContext(BaseModel):
    """Context for managing a user's conversation."""
    user_id: str
    session_start: datetime
    last_interaction: datetime
    turn_count: int = 0

    current_topic: Optional[str] = None
    conversation_history: List[ConversationTurn] = Field(default=[])

    user_preferences: dict = Field(default={})
    failure_context: Optional[dict] = None

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_123",
                "session_start": "2025-01-15T10:00:00Z",
                "last_interaction": "2025-01-15T10:05:00Z",
                "turn_count": 5,
                "current_topic": "voice_activity_detection",
                "conversation_history": [
                    {
                        "user_input": "What is VAD?",
                        "assistant_response": "VAD stands for Voice Activity Detection...",
                        "intent": "technical",
                        "timestamp": "2025-01-15T10:00:30Z",
                        "quality_score": 0.85
                    }
                ]
            }
        }


class FailureContext(BaseModel):
    """Context about failed interactions for recovery."""
    last_failure_reason: str
    failure_count: int = 0
    last_failed_intent: Optional[str] = None
    suggested_recovery: Optional[str] = None
    timestamp: datetime
