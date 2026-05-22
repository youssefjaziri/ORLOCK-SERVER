"""Integration tests for the full orchestration pipeline."""
import pytest
import asyncio
from datetime import datetime
from orlock.services.orchestration_service import OrchestrationService
from orlock.orchestration.context.manager import ContextManager
from orlock.intent.detector import IntentDetector
from orlock.quality.analyzer import SpeechQualityAnalyzer
from orlock.prompts.builder import PromptBuilder
from orlock.validation.validator import ResponseValidator
from orlock.schemas.metadata import SpeechMetadata
from orlock.services.llm_service import LLMService


class TestOrchestrationPipeline:
    """Test the full orchestration pipeline."""

    @pytest.mark.asyncio
    async def test_end_to_end_technical_question(self):
        """Test full pipeline with technical question."""
        # Initialize services
        context_manager = ContextManager()
        intent_detector = IntentDetector()
        quality_analyzer = SpeechQualityAnalyzer()
        prompt_builder = PromptBuilder()
        response_validator = ResponseValidator()
        llm_service = LLMService()

        orchestration = OrchestrationService(
            context_manager=context_manager,
            intent_detector=intent_detector,
            quality_analyzer=quality_analyzer,
            prompt_builder=prompt_builder,
            response_validator=response_validator,
            llm_service=llm_service
        )

        # Create test metadata
        metadata = SpeechMetadata(
            vad_confidence=0.87,
            speech_duration=2.5,
            silence_duration=0.5,
            prebuffer_duration=0.2,
            rms_energy_min=0.2,
            rms_energy_max=0.7,
            rms_energy_avg=0.45,
            amplitude_acceptance_rate=0.95,
            transcription_confidence=0.92
        )

        # Process transcription
        user_id = "test_user_123"
        transcription = "What is voice activity detection and how does it work?"

        result = await orchestration.process_transcription(
            user_id=user_id,
            transcription=transcription,
            metadata=metadata
        )

        # Verify result
        assert result.user_text == transcription
        assert len(result.llm_response) > 0
        assert result.intent is not None
        assert result.intent_confidence > 0
        assert result.speech_quality_score > 0
        assert result.processing_time_ms > 0
        assert len(result.pipeline_stages) > 0

        # Verify context was updated
        context = context_manager.get_context(user_id)
        assert context.turn_count == 1
        assert len(context.conversation_history) == 1

    @pytest.mark.asyncio
    async def test_context_persistence_across_turns(self):
        """Test that context persists across multiple turns."""
        context_manager = ContextManager()
        intent_detector = IntentDetector()
        quality_analyzer = SpeechQualityAnalyzer()
        prompt_builder = PromptBuilder()
        response_validator = ResponseValidator()
        llm_service = LLMService()

        orchestration = OrchestrationService(
            context_manager=context_manager,
            intent_detector=intent_detector,
            quality_analyzer=quality_analyzer,
            prompt_builder=prompt_builder,
            response_validator=response_validator,
            llm_service=llm_service
        )

        metadata = SpeechMetadata(
            vad_confidence=0.85,
            speech_duration=1.5,
            silence_duration=0.3,
            prebuffer_duration=0.1,
            transcription_confidence=0.9
        )

        user_id = "context_test_user"

        # First turn
        result1 = await orchestration.process_transcription(
            user_id=user_id,
            transcription="Tell me about VAD",
            metadata=metadata
        )

        assert result1.user_text == "Tell me about VAD"

        # Second turn
        result2 = await orchestration.process_transcription(
            user_id=user_id,
            transcription="How is it implemented?",
            metadata=metadata
        )

        assert result2.user_text == "How is it implemented?"

        # Verify context has both turns
        context = context_manager.get_context(user_id)
        assert context.turn_count == 2
        assert len(context.conversation_history) == 2
        assert context.conversation_history[0].user_input == "Tell me about VAD"
        assert context.conversation_history[1].user_input == "How is it implemented?"

    @pytest.mark.asyncio
    async def test_text_message_processing(self):
        """Test processing text messages (no speech metadata)."""
        context_manager = ContextManager()
        intent_detector = IntentDetector()
        quality_analyzer = SpeechQualityAnalyzer()
        prompt_builder = PromptBuilder()
        response_validator = ResponseValidator()
        llm_service = LLMService()

        orchestration = OrchestrationService(
            context_manager=context_manager,
            intent_detector=intent_detector,
            quality_analyzer=quality_analyzer,
            prompt_builder=prompt_builder,
            response_validator=response_validator,
            llm_service=llm_service
        )

        user_id = "text_user"
        text_input = "Hello, how are you?"

        result = await orchestration.process_message(
            user_id=user_id,
            user_text=text_input
        )

        assert result.user_text == text_input
        assert len(result.llm_response) > 0
        assert result.speech_quality_score == 1.0  # Perfect quality for text

    @pytest.mark.asyncio
    async def test_quality_influences_response(self):
        """Test that speech quality influences response generation."""
        context_manager = ContextManager()
        intent_detector = IntentDetector()
        quality_analyzer = SpeechQualityAnalyzer()
        prompt_builder = PromptBuilder()
        response_validator = ResponseValidator()
        llm_service = LLMService()

        orchestration = OrchestrationService(
            context_manager=context_manager,
            intent_detector=intent_detector,
            quality_analyzer=quality_analyzer,
            prompt_builder=prompt_builder,
            response_validator=response_validator,
            llm_service=llm_service
        )

        transcription = "What is VAD?"
        user_id = "quality_test"

        # High quality
        high_quality_metadata = SpeechMetadata(
            vad_confidence=0.95,
            speech_duration=3.0,
            silence_duration=0.5,
            prebuffer_duration=0.2,
            transcription_confidence=0.95
        )

        result_high = await orchestration.process_transcription(
            user_id=user_id,
            transcription=transcription,
            metadata=high_quality_metadata
        )

        # Low quality
        low_quality_metadata = SpeechMetadata(
            vad_confidence=0.25,
            speech_duration=0.3,
            silence_duration=1.0,
            prebuffer_duration=0.0,
            transcription_confidence=0.3
        )

        result_low = await orchestration.process_transcription(
            user_id=user_id,
            transcription=transcription,
            metadata=low_quality_metadata
        )

        # Verify quality scores differ
        assert result_high.speech_quality_score > result_low.speech_quality_score
        assert result_high.speech_quality_level != result_low.speech_quality_level


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
