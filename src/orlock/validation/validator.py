"""Response validation and quality checks."""
import logging
import re
from typing import NamedTuple
from ...schemas.intent import IntentCategory, IntentResult
from ...intent.categories import RESPONSE_LENGTH_EXPECTATIONS


logger = logging.getLogger(__name__)


class ValidationResult(NamedTuple):
    """Result of response validation."""
    is_valid: bool
    score: float  # 0.0 to 1.0
    issues: list
    suggestions: list


class ResponseValidator:
    """Validates LLM responses for quality and appropriateness."""

    def validate(self, response: str, intent: IntentResult) -> ValidationResult:
        """Validate response quality."""

        issues = []
        suggestions = []
        score = 1.0

        # Length check
        length_issue, length_suggestion, length_score = self._check_length(response, intent.category)
        if length_issue:
            issues.append(length_issue)
            suggestions.append(length_suggestion)
            score *= length_score

        # Coherence check
        coherence_issue, coherence_score = self._check_coherence(response)
        if coherence_issue:
            issues.append(coherence_issue)
            score *= coherence_score

        # Hallucination check
        hallucination_issue, hallucination_score = self._check_hallucination(response, intent.category)
        if hallucination_issue:
            issues.append(hallucination_issue)
            score *= hallucination_score

        # Safety check
        safety_issue = self._check_safety(response)
        if safety_issue:
            issues.append(safety_issue)
            score = 0.0

        # Format check
        format_issue, format_score = self._check_format(response, intent.category)
        if format_issue:
            issues.append(format_issue)
            score *= format_score

        is_valid = score > 0.6 and not safety_issue

        return ValidationResult(
            is_valid=is_valid,
            score=max(0.0, min(1.0, score)),
            issues=issues,
            suggestions=suggestions
        )

    def _check_length(self, response: str, intent: IntentCategory) -> tuple:
        """Check if response length is appropriate for intent."""

        min_tokens, max_tokens = RESPONSE_LENGTH_EXPECTATIONS.get(
            intent, (50, 300)
        )

        # Rough token estimate: ~4 chars per token
        estimated_tokens = len(response) // 4

        if estimated_tokens < min_tokens * 0.5:
            return (
                f"Response too short ({estimated_tokens} tokens, expected {min_tokens}+)",
                f"Expand the response with more detail",
                0.7
            )

        elif estimated_tokens > max_tokens * 2:
            return (
                f"Response too long ({estimated_tokens} tokens, expected <{max_tokens})",
                f"Condense the response",
                0.8
            )

        return None, None, 1.0

    def _check_coherence(self, response: str) -> tuple:
        """Check if response is coherent."""

        # Simple heuristics for coherence
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return "Response is empty", 0.0

        # Check for sentence structure
        avg_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0

        if avg_length < 2:
            return "Sentences too short, possibly incoherent", 0.6

        # Check for common coherence issues
        if response.count("[") != response.count("]"):
            return "Unmatched brackets, possible formatting issue", 0.8

        return None, 1.0

    def _check_hallucination(self, response: str, intent: IntentCategory) -> tuple:
        """Check for likely hallucinations."""

        # Technical responses shouldn't have obviously made-up details
        if intent == IntentCategory.TECHNICAL:
            hallucination_markers = [
                "I just invented",
                "I made up",
                "I'm not sure but",
                "probably exists",
                "maybe there's"
            ]

            for marker in hallucination_markers:
                if marker.lower() in response.lower():
                    return "Possible hallucination marker detected", 0.5

        # Check for contradictions within response
        if "however," in response.lower() or "but" in response.lower():
            # Simple check: conflicting statements
            before_but = response.lower().split("but")[0]
            after_but = response.lower().split("but")[1] if len(response.split("but")) > 1 else ""

            if "always" in before_but and "never" in after_but:
                return "Internal contradiction detected", 0.7

        return None, 1.0

    def _check_safety(self, response: str) -> str:
        """Check for unsafe or harmful content."""

        unsafe_patterns = [
            r"execute immediately",
            r"delete everything",
            r"harmful",
            r"dangerous"
        ]

        for pattern in unsafe_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return f"Potentially unsafe content detected: {pattern}"

        return None

    def _check_format(self, response: str, intent: IntentCategory) -> tuple:
        """Check if response follows expected format for intent."""

        if intent == IntentCategory.COMMAND:
            # Commands should have clear action items
            if not any(word in response.lower() for word in ["do", "execute", "run", "perform", "click"]):
                return "Command response lacks action items", 0.8

        elif intent == IntentCategory.SUMMARIZATION:
            # Summaries should be noticeably shorter than original
            if len(response) > 500:
                return "Summary is too long", 0.8

        elif intent == IntentCategory.TECHNICAL:
            # Technical responses should have structure
            if response.count("\n") < 2:
                return "Technical response lacks structure", 0.7

        return None, 1.0


class FallbackHandler:
    """Handles fallback responses for validation failures."""

    FALLBACK_RESPONSES = {
        "too_short": "I apologize, but I don't have enough information to provide a complete answer. Could you please provide more context or rephrase your question?",
        "too_long": "Let me provide a more concise answer: [I'll be more brief]",
        "incoherent": "I apologize, my response wasn't clear. Let me try again with a better explanation.",
        "hallucination": "I'm not certain about that detail. Let me stick to what I'm confident about.",
        "safety_issue": "I can't provide that response. Is there something else I can help with?",
        "format_issue": "Let me restructure that response for clarity.",
        "generic": "I apologize, but I had trouble with that response. Could you try asking again?",
    }

    @staticmethod
    def get_fallback(reason: str) -> str:
        """Get appropriate fallback response."""
        return FallbackHandler.FALLBACK_RESPONSES.get(reason, FallbackHandler.FALLBACK_RESPONSES["generic"])
