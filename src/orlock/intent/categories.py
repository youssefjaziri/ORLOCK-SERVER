"""Intent categories and definitions for the orchestration layer."""
from enum import Enum


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


RECOMMENDED_TEMPERATURES = {
    IntentCategory.COMMAND: 0.1,
    IntentCategory.TECHNICAL: 0.15,
    IntentCategory.EMERGENCY: 0.1,
    IntentCategory.ACKNOWLEDGEMENT: 0.0,
    IntentCategory.SYSTEM_CONTROL: 0.1,
    IntentCategory.EXPLANATION: 0.25,
    IntentCategory.QUESTION: 0.2,
    IntentCategory.CLARIFICATION: 0.2,
    IntentCategory.CONVERSATIONAL: 0.3,
    IntentCategory.GREETING: 0.35,
    IntentCategory.NAVIGATION: 0.15,
    IntentCategory.SUMMARIZATION: 0.15,
    IntentCategory.TASK_EXECUTION: 0.2,
}


RESPONSE_LENGTH_EXPECTATIONS = {
    IntentCategory.ACKNOWLEDGEMENT: (10, 30),
    IntentCategory.GREETING: (20, 100),
    IntentCategory.COMMAND: (30, 150),
    IntentCategory.CLARIFICATION: (50, 200),
    IntentCategory.NAVIGATION: (50, 250),
    IntentCategory.TECHNICAL: (100, 500),
    IntentCategory.EXPLANATION: (150, 600),
    IntentCategory.QUESTION: (75, 400),
    IntentCategory.SUMMARIZATION: (75, 300),
    IntentCategory.CONVERSATIONAL: (50, 300),
    IntentCategory.SYSTEM_CONTROL: (50, 200),
    IntentCategory.EMERGENCY: (20, 150),
    IntentCategory.TASK_EXECUTION: (200, 800),
}
