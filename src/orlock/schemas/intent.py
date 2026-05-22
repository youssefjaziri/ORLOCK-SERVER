from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Tuple


class IntentCategory(str, Enum):
    """13 intent categories for intelligent response generation."""
    CONVERSATIONAL = "conversational"
    QUESTION = "question"
    COMMAND = "command"
    SUMMARIZATION = "summarization"
    EXPLANATION = "explanation"
    TASK_EXECUTION = "task_execution"
    GREETING = "greeting"
    CLARIFICATION = "clarification"
    EMERGENCY = "emergency"
    ACKNOWLEDGEMENT = "acknowledgement"
    SYSTEM_CONTROL = "system_control"
    NAVIGATION = "navigation"
    TECHNICAL = "technical"


INTENT_DESCRIPTIONS = {
    IntentCategory.CONVERSATIONAL: "General conversation, small talk, social interaction",
    IntentCategory.QUESTION: "Factual or clarifying questions",
    IntentCategory.COMMAND: "Direct action requests, instructions",
    IntentCategory.SUMMARIZATION: "Request to condense or summarize information",
    IntentCategory.EXPLANATION: "Request for detailed explanation or breakdown",
    IntentCategory.TASK_EXECUTION: "Multi-step task requests, project work",
    IntentCategory.GREETING: "Hello, goodbye, introduction",
    IntentCategory.CLARIFICATION: "Requests for clarification on previous statements",
    IntentCategory.EMERGENCY: "Urgent, critical, or time-sensitive requests",
    IntentCategory.ACKNOWLEDGEMENT: "Simple confirmations, affirmations",
    IntentCategory.SYSTEM_CONTROL: "System settings, configuration, administration",
    IntentCategory.NAVIGATION: "Route finding, location information, directions",
    IntentCategory.TECHNICAL: "Technical questions, debugging, implementation help",
}


class IntentResult(BaseModel):
    """Result of intent classification."""
    category: IntentCategory
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")
    reasoning: str = Field(..., description="Why this intent was selected")

    related_categories: List[Tuple[IntentCategory, float]] = Field(
        default=[],
        description="Other possible intents with scores"
    )
    suggested_response_style: str = Field(
        ...,
        description="Tone/style guidance for response generation"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "category": "technical",
                "confidence": 0.92,
                "reasoning": "User asked a technical question about VAD implementation",
                "related_categories": [
                    ["question", 0.8],
                    ["explanation", 0.75]
                ],
                "suggested_response_style": "professional, code-focused, structured"
            }
        }
