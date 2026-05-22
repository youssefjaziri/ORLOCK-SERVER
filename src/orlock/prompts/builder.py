"""Dynamic prompt engineering based on intent, context, and metadata."""
import logging
from pathlib import Path
from typing import Tuple
from ...schemas.intent import IntentResult, IntentCategory
from ...schemas.metadata import SpeechMetadata, SpeechQualityScore
from ...orchestration.context.models import ConversationContext


logger = logging.getLogger(__name__)


class PromptBuilder:
    """Builds dynamic prompts based on intent, context, and speech quality."""

    def __init__(self, templates_dir: str = "src/orlock/prompts/templates"):
        self.templates_dir = Path(templates_dir)
        self._template_cache = {}

    def build_system_prompt(self,
                           intent: IntentResult,
                           metadata: SpeechMetadata,
                           context: ConversationContext,
                           quality: SpeechQualityScore) -> str:
        """Build dynamic system prompt based on intent and context."""

        base_template = self._load_template(intent.category)

        # Add quality-based modifications
        quality_modifiers = self._get_quality_modifiers(quality)

        # Add context-based modifications
        context_modifiers = self._get_context_modifiers(context)

        # Add speech metadata modifications
        metadata_modifiers = self._get_metadata_modifiers(metadata, quality)

        full_prompt = base_template.strip()
        if quality_modifiers:
            full_prompt += f"\n\n{quality_modifiers}"
        if context_modifiers:
            full_prompt += f"\n\n{context_modifiers}"
        if metadata_modifiers:
            full_prompt += f"\n\n{metadata_modifiers}"

        return full_prompt

    def build_full_prompt(self,
                         user_input: str,
                         intent: IntentResult,
                         metadata: SpeechMetadata,
                         context: ConversationContext,
                         quality: SpeechQualityScore) -> Tuple[str, str]:
        """Build both system and user prompts."""

        system_prompt = self.build_system_prompt(intent, metadata, context, quality)

        # Build user-facing prompt with context
        context_str = self._build_context_string(context, limit=5)

        user_prompt = f"{context_str}{user_input}" if context_str else user_input

        return system_prompt, user_prompt

    def _load_template(self, category: IntentCategory) -> str:
        """Load prompt template for intent category."""

        # Check cache
        if category in self._template_cache:
            return self._template_cache[category]

        template_file = self.templates_dir / f"{category.value}.md"

        try:
            if template_file.exists():
                with open(template_file, "r") as f:
                    template = f.read()
                self._template_cache[category] = template
                return template
            else:
                logger.warning(f"Template not found for {category.value}, using default")
                return self._get_default_template()
        except Exception as e:
            logger.error(f"Failed to load template for {category}: {e}")
            return self._get_default_template()

    def _get_quality_modifiers(self, quality: SpeechQualityScore) -> str:
        """Get modifications based on speech quality."""

        if quality.quality_level == "excellent":
            return "The audio quality is excellent. Generate a detailed, confident response."

        elif quality.quality_level == "good":
            return "The audio quality is good. Provide a clear, direct response."

        elif quality.quality_level == "fair":
            return "Audio quality is fair. Be slightly cautious. If there's any ambiguity, ask for clarification."

        elif quality.quality_level == "poor":
            return "Audio quality is poor. Ask the user to repeat or speak more clearly."

        else:  # unintelligible
            return "Audio quality is very poor. Request the user to repeat clearly."

    def _get_context_modifiers(self, context: ConversationContext) -> str:
        """Get modifications based on conversation context."""

        modifiers = []

        if context.turn_count > 10:
            modifiers.append("This is an extended conversation. Maintain continuity with the discussion so far.")

        if context.current_topic:
            modifiers.append(f"Current discussion topic: {context.current_topic}")

        if len(context.conversation_history) > 0:
            last_intent = context.conversation_history[-1].intent
            if last_intent:
                modifiers.append(f"The previous message was about {last_intent}. Maintain topic coherence.")

        if context.failure_context:
            modifiers.append("A previous response didn't work well. Be more careful this time.")

        return "\n".join(modifiers) if modifiers else ""

    def _get_metadata_modifiers(self, metadata: SpeechMetadata, quality: SpeechQualityScore) -> str:
        """Get modifications based on speech metadata."""

        modifiers = []

        if metadata.speech_duration < 0.5:
            modifiers.append("This was a very short utterance. Keep response concise and direct.")

        elif metadata.speech_duration > 5.0:
            modifiers.append("This was a long utterance. The user provided substantial input. Address the full scope.")

        if metadata.vad_confidence < 0.5:
            modifiers.append("Voice detection confidence is low. Be cautious in interpretation.")

        if metadata.silence_duration > 2.0:
            modifiers.append("There was a long pause. The user might be thinking. Answer thoughtfully.")

        return "\n".join(modifiers) if modifiers else ""

    def _build_context_string(self, context: ConversationContext, limit: int = 5) -> str:
        """Build context string from recent conversation history."""

        if not context.conversation_history:
            return ""

        recent_turns = context.conversation_history[-limit:]

        context_lines = []
        for turn in recent_turns:
            context_lines.append(f"User: {turn.user_input[:100]}")
            context_lines.append(f"Assistant: {turn.assistant_response[:100]}")

        if not context_lines:
            return ""

        context_text = "\n".join(context_lines)
        return f"Previous conversation:\n{context_text}\n\nNew user input: "

    def _get_default_template(self) -> str:
        """Get default template when specific one is not found."""
        return (
            "You are a helpful assistant. Respond directly to the user's input. "
            "Be clear, accurate, and conversational. "
            "If the user asks for clarification, provide it carefully."
        )
