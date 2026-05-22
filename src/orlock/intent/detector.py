"""Intent detection using LLM classification."""
import logging
import json
import asyncio
from typing import Optional, Dict
from functools import lru_cache
from ..schemas.intent import IntentResult, IntentCategory
from ..schemas.metadata import SpeechMetadata
from ..intent.categories import INTENT_DESCRIPTIONS
from ...services.llm_service import LLMService


logger = logging.getLogger(__name__)


INTENT_CLASSIFIER_PROMPT = """You are an expert at classifying user intents. Analyze the user's input and determine their primary intent.

Available intent categories:
{categories}

User input: "{user_input}"

Respond with a JSON object containing:
- "intent": one of the categories above
- "confidence": float between 0 and 1
- "reasoning": brief explanation
- "related_intents": list of [category, confidence] pairs for runner-ups

Example:
{{"intent": "technical", "confidence": 0.92, "reasoning": "User asked a technical question", "related_intents": [["question", 0.8]]}}

IMPORTANT: You MUST respond with ONLY valid JSON, no other text."""


class IntentDetector:
    """Detects user intent from text input."""

    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or LLMService()
        self._intent_cache: Dict[str, IntentResult] = {}
        self._cache_max_size = 1000

    async def detect(self, text: str, metadata: Optional[SpeechMetadata] = None) -> IntentResult:
        """Detect intent from text input."""
        # Check cache first
        cached = self._get_cached_intent(text)
        if cached:
            logger.debug(f"Intent from cache for: {text[:50]}")
            return cached

        try:
            intent_result = await self._classify_with_llm(text)
            self._cache_intent(text, intent_result)
            return intent_result
        except Exception as e:
            logger.error(f"Intent detection failed: {e}. Falling back to CONVERSATIONAL")
            return IntentResult(
                category=IntentCategory.CONVERSATIONAL,
                confidence=0.5,
                reasoning="Intent detection failed, defaulting to conversational",
                related_categories=[],
                suggested_response_style="neutral, helpful"
            )

    async def _classify_with_llm(self, text: str) -> IntentResult:
        """Use LLM to classify intent."""
        # Build category descriptions
        categories_text = "\n".join(
            f"- {cat.value}: {INTENT_DESCRIPTIONS[cat]}"
            for cat in IntentCategory
        )

        prompt = INTENT_CLASSIFIER_PROMPT.format(
            categories=categories_text,
            user_input=text
        )

        system_prompt = "You are a precise intent classification system. Respond only with valid JSON."

        response = await asyncio.to_thread(
            self.llm_service.message_to_llm_text,
            prompt,
            system_prompt,
            temperature=0.1
        )

        return self._parse_intent_response(response)

    def _parse_intent_response(self, response: str) -> IntentResult:
        """Parse LLM response to IntentResult."""
        try:
            data = json.loads(response.strip())

            intent_cat = IntentCategory(data.get("intent", "conversational"))
            confidence = float(data.get("confidence", 0.5))
            reasoning = data.get("reasoning", "")

            related = []
            for cat_name, score in data.get("related_intents", []):
                try:
                    related.append((IntentCategory(cat_name), float(score)))
                except (ValueError, KeyError):
                    pass

            style = self._get_suggested_style(intent_cat, confidence)

            return IntentResult(
                category=intent_cat,
                confidence=min(1.0, max(0.0, confidence)),
                reasoning=reasoning,
                related_categories=related,
                suggested_response_style=style
            )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse intent response: {e}. Response: {response}")
            return IntentResult(
                category=IntentCategory.CONVERSATIONAL,
                confidence=0.3,
                reasoning="Failed to parse intent, defaulting to conversational",
                related_categories=[],
                suggested_response_style="neutral"
            )

    def _get_suggested_style(self, intent: IntentCategory, confidence: float) -> str:
        """Get suggested response style for intent."""
        style_map = {
            IntentCategory.TECHNICAL: "professional, code-focused, structured",
            IntentCategory.COMMAND: "action-oriented, concise, directive",
            IntentCategory.QUESTION: "informative, clear, direct",
            IntentCategory.EXPLANATION: "detailed, educational, step-by-step",
            IntentCategory.GREETING: "warm, friendly, personable",
            IntentCategory.EMERGENCY: "urgent, focused, solution-oriented",
            IntentCategory.ACKNOWLEDGEMENT: "brief, affirmative, supportive",
            IntentCategory.CLARIFICATION: "explicit, thorough, unambiguous",
            IntentCategory.CONVERSATIONAL: "natural, conversational, engaging",
            IntentCategory.SUMMARIZATION: "concise, bullet-point, organized",
            IntentCategory.TASK_EXECUTION: "structured, methodical, comprehensive",
            IntentCategory.SYSTEM_CONTROL: "precise, technical, procedural",
            IntentCategory.NAVIGATION: "clear, sequential, oriented",
        }

        base_style = style_map.get(intent, "neutral, helpful")

        if confidence < 0.6:
            base_style += ", ask for clarification if needed"

        return base_style

    def _get_cached_intent(self, text: str) -> Optional[IntentResult]:
        """Get cached intent if available."""
        key = self._make_cache_key(text)
        return self._intent_cache.get(key)

    def _cache_intent(self, text: str, result: IntentResult):
        """Cache intent result."""
        if len(self._intent_cache) >= self._cache_max_size:
            # Clear cache if too large (simple FIFO)
            first_key = next(iter(self._intent_cache))
            del self._intent_cache[first_key]

        key = self._make_cache_key(text)
        self._intent_cache[key] = result

    @staticmethod
    def _make_cache_key(text: str) -> str:
        """Create cache key from text (normalized)."""
        return text.strip().lower()[:200]
