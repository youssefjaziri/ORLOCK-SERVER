"""Orchestrated audio endpoint - intelligent transcription and response generation."""
import logging
from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from ...schemas.metadata import SpeechMetadata
from ...schemas.orchestration import OrchestrationResponse
from ...services.orchestration_service import OrchestrationService
from ...services.transcription_service import TranscriptionService


logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services
orchestration_service = OrchestrationService()
transcription_service = TranscriptionService()


@router.post("/orchestrated/audio", response_model=OrchestrationResponse)
async def orchestrated_audio(
    user_id: str = Form(...),
    audio: UploadFile = File(...),
    metadata_json: str = Form(...),
    system_prompt: str = Form(None)
):
    """
    Orchestrated audio endpoint with intelligent intent detection and response generation.

    Accepts:
    - user_id: User identifier
    - audio: Audio file (WAV)
    - metadata_json: JSON-encoded SpeechMetadata with VAD and audio quality metrics
    - system_prompt: Optional custom system prompt

    Returns:
    - OrchestrationResponse with intent, quality score, and response
    """

    try:
        # Validate and parse metadata
        try:
            metadata = SpeechMetadata.model_validate_json(metadata_json)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid metadata JSON: {e}")

        # Validate audio file
        if not audio.filename:
            raise HTTPException(status_code=400, detail="Audio file is required")

        audio_content = await audio.read()
        if not audio_content:
            raise HTTPException(status_code=400, detail="Audio file is empty")

        # Transcribe audio
        logger.debug(f"Transcribing audio for user {user_id}")
        transcript = await transcription_service.transcribe_audio(audio_content)

        if not transcript:
            raise HTTPException(status_code=500, detail="Transcription failed")

        logger.debug(f"Transcribed: {transcript[:50]}")

        # Process through orchestration pipeline
        logger.debug("Starting orchestration pipeline")
        result = await orchestration_service.process_transcription(
            user_id=user_id,
            transcription=transcript,
            metadata=metadata,
            user_system_prompt=system_prompt
        )

        logger.info(
            f"Orchestrated response for {user_id}: "
            f"intent={result.intent.value}, "
            f"quality={result.speech_quality_level}, "
            f"time={result.processing_time_ms:.0f}ms"
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in orchestrated audio endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
