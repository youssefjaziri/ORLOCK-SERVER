"""High-level orchestration service that coordinates the intelligent response generation pipeline."""
import logging
import time
from typing import Optional
from ...schemas.metadata import SpeechMetadata
from ...schemas.orchestration import OrchestrationResponse
from ...schemas.intent import IntentCategory
from ...intent.detector import IntentDetector
from ...quality.analyzer import SpeechQualityAnalyzer
from ...prompts.builder import PromptBuilder
from ...validation.validator import ResponseValidator, FallbackHandler
from ...orchestration.context.manager import ContextManager
from ...intent.categories import RECOMMENDED_TEMPERATURES
from ...services.llm_service import LLMService


logger = logging.getLogger(__name__)


class OrchestrationService:
    """Main orchestration service for intelligent response generation."""

    def __init__(self,
                 context_manager: Optional[ContextManager] = None,
                 intent_detector: Optional[IntentDetector] = None,
                 quality_analyzer: Optional[SpeechQualityAnalyzer] = None,
                 prompt_builder: Optional[PromptBuilder] = None,
                 response_validator: Optional[ResponseValidator] = None,
                 llm_service: Optional[LLMService] = None):

        self.context_manager = context_manager or ContextManager()
        self.intent_detector = intent_detector or IntentDetector()
        self.quality_analyzer = quality_analyzer or SpeechQualityAnalyzer()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.response_validator = response_validator or ResponseValidator()
        self.llm_service = llm_service or LLMService()

    async def process_transcription(self,
                                    user_id: str,
                                    transcription: str,
                                    metadata: SpeechMetadata,
                                    user_system_prompt: Optional[str] = None) -> OrchestrationResponse:
        """Process a transcription through the full orchestration pipeline."""

        pipeline_times = {}
        start_time = time.time()

        try:
            # Stage 1: Intent Detection
            stage_start = time.time()
            intent = await self.intent_detector.detect(transcription, metadata)
            pipeline_times["intent_detection"] = (time.time() - stage_start) * 1000

            # Stage 2: Context Retrieval
            stage_start = time.time()
            context = self.context_manager.get_context(user_id)
            pipeline_times["context_retrieval"] = (time.time() - stage_start) * 1000

            # Stage 3: Quality Analysis
            stage_start = time.time()
            quality = self.quality_analyzer.analyze(metadata)
            pipeline_times["quality_analysis"] = (time.time() - stage_start) * 1000

            # Stage 4: Prompt Building
            stage_start = time.time()
            system_prompt, user_prompt = self.prompt_builder.build_full_prompt(
                transcription, intent, metadata, context, quality
            )

            # Override with user-provided system prompt if given
            if user_system_prompt:
                system_prompt = user_system_prompt

            pipeline_times["prompt_building"] = (time.time() - stage_start) * 1000

            # Stage 5: LLM Generation
            stage_start = time.time()
            temperature = self._select_temperature(intent, quality)

            llm_response = await self._call_llm(user_prompt, system_prompt, temperature)
            pipeline_times["llm_generation"] = (time.time() - stage_start) * 1000

            # Stage 6: Response Validation
            stage_start = time.time()
            validation = self.response_validator.validate(llm_response, intent)

            if not validation.is_valid:
                llm_response = self._apply_fallback(validation, intent)

            pipeline_times["response_validation"] = (time.time() - stage_start) * 1000

            # Update context
            self.context_manager.add_turn(
                user_id,
                transcription,
                llm_response,
                intent=intent.category.value,
                quality_score=quality.overall_score
            )

            total_time = (time.time() - start_time) * 1000

            return OrchestrationResponse(
                user_text=transcription,
                llm_response=llm_response,
                intent=intent.category,
                intent_confidence=intent.confidence,
                speech_quality_score=quality.overall_score,
                speech_quality_level=quality.quality_level,
                metadata=metadata,
                processing_time_ms=total_time,
                pipeline_stages=pipeline_times
            )

        except Exception as e:
            logger.error(f"Error in orchestration pipeline: {e}", exc_info=True)

            # Return graceful fallback
            return OrchestrationResponse(
                user_text=transcription,
                llm_response=FallbackHandler.get_fallback("generic"),
                intent=IntentCategory.CONVERSATIONAL,
                intent_confidence=0.0,
                speech_quality_score=0.0,
                metadata=metadata,
                processing_time_ms=(time.time() - start_time) * 1000,
                pipeline_stages=pipeline_times
            )

    async def process_message(self,
                             user_id: str,
                             user_text: str,
                             user_system_prompt: Optional[str] = None) -> OrchestrationResponse:
        """Process a text message (no speech metadata)."""

        # Create minimal metadata for text-only input
        metadata = SpeechMetadata(
            vad_confidence=1.0,
            speech_duration=1.0,
            silence_duration=0.0,
            prebuffer_duration=0.0,
            transcription_confidence=1.0
        )

        return await self.process_transcription(user_id, user_text, metadata, user_system_prompt)

    def _select_temperature(self, intent, quality) -> float:
        """Select appropriate temperature based on intent and quality."""

        base_temp = RECOMMENDED_TEMPERATURES.get(intent.category, 0.2)

        # Reduce temperature for low-quality speech
        if quality.quality_level in ["poor", "unintelligible"]:
            base_temp = min(base_temp, 0.1)

        # Low confidence intent → be more conservative
        if intent.confidence < 0.5:
            base_temp = min(base_temp, 0.15)

        return base_temp

    async def _call_llm(self, prompt: str, system: str, temperature: float) -> str:
        """Call LLM with given parameters."""
        return await self._run_async(
            self.llm_service.message_to_llm_text,
            prompt,
            system,
            temperature
        )

    async def _run_async(self, func, *args, **kwargs):
        """Run synchronous function asynchronously."""
        import asyncio
        return await asyncio.to_thread(func, *args, **kwargs)

    def _apply_fallback(self, validation, intent) -> str:
        """Apply fallback response based on validation results."""

        if not validation.issues:
            return FallbackHandler.get_fallback("generic")

        primary_issue = validation.issues[0]

        if "too short" in primary_issue.lower():
            return FallbackHandler.get_fallback("too_short")
        elif "too long" in primary_issue.lower():
            return FallbackHandler.get_fallback("too_long")
        elif "incoherent" in primary_issue.lower():
            return FallbackHandler.get_fallback("incoherent")
        elif "hallucination" in primary_issue.lower():
            return FallbackHandler.get_fallback("hallucination")
        elif "format" in primary_issue.lower():
            return FallbackHandler.get_fallback("format_issue")
        else:
            return FallbackHandler.get_fallback("generic")
