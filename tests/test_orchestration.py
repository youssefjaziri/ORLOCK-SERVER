"""Unit tests for orchestration layer components."""
import pytest
import asyncio
from datetime import datetime
from orlock.intent.detector import IntentDetector
from orlock.intent.categories import IntentCategory
from orlock.quality.analyzer import SpeechQualityAnalyzer
from orlock.schemas.metadata import SpeechMetadata
from orlock.orchestration.context.manager import ContextManager
from orlock.prompts.builder import PromptBuilder
from orlock.validation.validator import ResponseValidator


class TestIntentDetector:
    """Test intent detection."""

    @pytest.mark.asyncio
    async def test_detect_technical_question(self):
        """Test detecting technical questions."""
        detector = IntentDetector()
        result = await detector.detect("What is voice activity detection?")

        assert result.category in [IntentCategory.TECHNICAL, IntentCategory.QUESTION]
        assert result.confidence > 0.5

    @pytest.mark.asyncio
    async def test_detect_greeting(self):
        """Test detecting greetings."""
        detector = IntentDetector()
        result = await detector.detect("Hello, how are you?")

        assert result.category == IntentCategory.GREETING or result.confidence < 0.6
        assert result.reasoning

    @pytest.mark.asyncio
    async def test_detect_command(self):
        """Test detecting commands."""
        detector = IntentDetector()
        result = await detector.detect("Start the audio processing")

        assert result.category in [IntentCategory.COMMAND, IntentCategory.TASK_EXECUTION]
        assert result.confidence > 0.4


class TestSpeechQualityAnalyzer:
    """Test speech quality analysis."""

    def test_excellent_quality(self):
        """Test excellent quality detection."""
        analyzer = SpeechQualityAnalyzer()
        metadata = SpeechMetadata(
            vad_confidence=0.95,
            speech_duration=3.5,
            silence_duration=0.5,
            prebuffer_duration=0.2,
            rms_energy_avg=0.5
        )

        quality = analyzer.analyze(metadata)
        assert quality.quality_level == "excellent"
        assert quality.overall_score > 0.8

    def test_poor_quality(self):
        """Test poor quality detection."""
        analyzer = SpeechQualityAnalyzer()
        metadata = SpeechMetadata(
            vad_confidence=0.2,
            speech_duration=0.3,
            silence_duration=1.0,
            prebuffer_duration=0.0
        )

        quality = analyzer.analyze(metadata)
        assert quality.quality_level in ["poor", "unintelligible"]
        assert quality.overall_score < 0.5

    def test_fair_quality(self):
        """Test fair quality detection."""
        analyzer = SpeechQualityAnalyzer()
        metadata = SpeechMetadata(
            vad_confidence=0.6,
            speech_duration=1.2,
            silence_duration=0.5,
            prebuffer_duration=0.1
        )

        quality = analyzer.analyze(metadata)
        assert quality.quality_level in ["fair", "good"]


class TestContextManager:
    """Test context management."""

    def test_create_context(self):
        """Test context creation."""
        manager = ContextManager()
        context = manager.get_context("user_123")

        assert context.user_id == "user_123"
        assert context.turn_count == 0
        assert len(context.conversation_history) == 0

    def test_add_turn(self):
        """Test adding conversation turns."""
        manager = ContextManager()
        manager.add_turn("user_123", "Hello", "Hi there!", intent="greeting")

        context = manager.get_context("user_123")
        assert context.turn_count == 1
        assert len(context.conversation_history) == 1
        assert context.conversation_history[0].user_input == "Hello"

    def test_history_limit(self):
        """Test conversation history is limited."""
        manager = ContextManager(max_turns=5)

        for i in range(10):
            manager.add_turn("user_123", f"Message {i}", f"Response {i}")

        context = manager.get_context("user_123")
        assert len(context.conversation_history) == 5

    def test_get_history_for_llm(self):
        """Test getting formatted chat history."""
        manager = ContextManager()
        manager.add_turn("user_123", "What is VAD?", "VAD is...", intent="technical")
        manager.add_turn("user_123", "How does it work?", "It works by...", intent="technical")

        history = manager.get_history("user_123")
        assert len(history) == 4
        assert history[0].role == "user"
        assert history[1].role == "assistant"


class TestPromptBuilder:
    """Test dynamic prompt building."""

    def test_build_technical_prompt(self):
        """Test building prompt for technical intent."""
        builder = PromptBuilder()
        from orlock.schemas.intent import IntentResult
        from orlock.orchestration.context.models import ConversationContext

        intent = IntentResult(
            category=IntentCategory.TECHNICAL,
            confidence=0.9,
            reasoning="Technical question",
            related_categories=[],
            suggested_response_style="professional"
        )

        metadata = SpeechMetadata(
            vad_confidence=0.85,
            speech_duration=2.0,
            silence_duration=0.5,
            prebuffer_duration=0.1
        )

        from orlock.quality.analyzer import SpeechQualityAnalyzer
        analyzer = SpeechQualityAnalyzer()
        quality = analyzer.analyze(metadata)

        context = ConversationContext(
            user_id="user_123",
            session_start=datetime.now(),
            last_interaction=datetime.now()
        )

        system_prompt = builder.build_system_prompt(intent, metadata, context, quality)
        assert len(system_prompt) > 0
        assert "technical" in system_prompt.lower() or "professional" in system_prompt.lower()

    def test_quality_modifier_excellent(self):
        """Test quality modifiers for excellent speech."""
        builder = PromptBuilder()
        from orlock.schemas.metadata import SpeechQualityScore

        quality = SpeechQualityScore(
            quality_level="excellent",
            overall_score=0.95,
            vad_score=0.95,
            duration_score=0.95,
            energy_score=0.9,
            confidence_score=0.95,
            reasoning="Excellent quality"
        )

        modifier = builder._get_quality_modifiers(quality)
        assert "excellent" in modifier.lower() or "detailed" in modifier.lower()


class TestResponseValidator:
    """Test response validation."""

    def test_validate_good_response(self):
        """Test validating a good response."""
        validator = ResponseValidator()
        response = "Voice Activity Detection (VAD) is a technique that detects the presence of speech in audio signals."

        intent = IntentResult(
            category=IntentCategory.TECHNICAL,
            confidence=0.9,
            reasoning="Technical question",
            related_categories=[],
            suggested_response_style="professional"
        )

        result = validator.validate(response, intent)
        assert result.is_valid is True
        assert result.score > 0.6

    def test_validate_too_short_response(self):
        """Test detecting too-short responses."""
        validator = ResponseValidator()
        response = "Yes."

        intent = IntentResult(
            category=IntentCategory.EXPLANATION,
            confidence=0.9,
            reasoning="Explanation request",
            related_categories=[],
            suggested_response_style="detailed"
        )

        result = validator.validate(response, intent)
        assert "too short" in " ".join(result.issues).lower() or result.score < 0.8

    def test_fallback_response(self):
        """Test getting fallback responses."""
        from orlock.validation.validator import FallbackHandler

        fallback = FallbackHandler.get_fallback("too_short")
        assert len(fallback) > 0
        assert isinstance(fallback, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
