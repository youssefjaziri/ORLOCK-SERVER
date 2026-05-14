from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
from pathlib import Path
from uuid import uuid4
import logging
import sys
import time

from orlock.schemas.user_message import UserMessageOut
from orlock.schemas.llm import MessageToLLMRequest
from orlock.services.llm_service import LLMService
from orlock.services.transcription_service import TranscriptionService

# Import monitoring with better error handling
try:
    # Try absolute path first
    sys.path.insert(0, '/home/youssef/PFE2/monitoring')
    from pipeline_logger import get_logger, PipelineStage
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    get_logger = None
    PipelineStage = None

# Import Whisper for transcription
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    WhisperModel = None

router = APIRouter(tags=["audio"])
logger = logging.getLogger(__name__)

# pasta onde vais guardar temporariamente
TEMP_AUDIO_DIR = Path("tempaudio")

# Initialize services
_whisper_model = None
_transcription_service = TranscriptionService()

def get_whisper_model():
    """Lazy load Whisper model on first use."""
    global _whisper_model
    if _whisper_model is None and WHISPER_AVAILABLE:
        logger.info("Loading Whisper model...")
        _whisper_model = WhisperModel("base", device="cpu")
        logger.info("Whisper model loaded successfully")
    return _whisper_model

def transcribe_audio(audio_path: Path) -> str:
    """Transcribe audio file using Whisper."""
    if not WHISPER_AVAILABLE:
        logger.warning("Whisper not available, using placeholder")
        return f"[TRANSCRIPTION_UNAVAILABLE] Could not transcribe audio at {audio_path}"

    try:
        model = get_whisper_model()
        if model is None:
            return f"[TRANSCRIPTION_ERROR] Whisper model failed to load"

        logger.info(f"Transcribing audio: {audio_path}")
        # transcribe() returns a tuple: (segments_generator, transcription_info)
        segments, info = model.transcribe(str(audio_path), language="en")

        # Convert generator to list and extract text from each segment
        transcript = " ".join([
            segment.text.strip() for segment in segments
        ])

        if not transcript.strip():
            transcript = "[TRANSCRIPTION_EMPTY] No speech detected in audio"
            logger.warning(f"No speech detected in {audio_path}")

        logger.info(f"Transcription complete: {transcript[:100]}...")
        return transcript

    except Exception as e:
        error_msg = f"[TRANSCRIPTION_ERROR] {str(e)}"
        logger.error(f"Transcription failed for {audio_path}: {str(e)}")
        return error_msg


@router.post("/userAudio", response_model=UserMessageOut)
async def user_audio(
    user_id: str = Form(...),
    audio: UploadFile = File(...),
    system: Optional[str] = Form(None),
    temperature: float = Form(0.2),
):
    monitor_logger = get_logger() if MONITORING_AVAILABLE else None
    start_time = time.time()

    try:
        if monitor_logger:
            monitor_logger.info("API", f"Received audio from {user_id}")

        # 1) garantir que a pasta existe
        TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

        # 2) ler bytes do ficheiro
        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Ficheiro de áudio vazio.")

        # 3) escolher nome único e gravar em disco
        original_suffix = Path(audio.filename or "").suffix.lower()
        suffix = original_suffix if original_suffix else ".wav"

        safe_name = f"{user_id}_{uuid4().hex}{suffix}"
        save_path = TEMP_AUDIO_DIR / safe_name

        with open(save_path, "wb") as f:
            f.write(audio_bytes)

        logger.info(f"Audio saved: {save_path} ({len(audio_bytes)} bytes)")
        if monitor_logger:
            monitor_logger.debug("API", f"Audio saved ({len(audio_bytes)} bytes)")

        # 4) TRANSCRIÇÃO - Use actual Whisper transcription
        if monitor_logger:
            monitor_logger.stage_start("TRANSCRIBING")

        transcript_text = transcribe_audio(save_path)

        if monitor_logger:
            monitor_logger.stage_end("TRANSCRIBING", PipelineStage.TRANSCRIBING)

        # 5) enviar para LLM
        if monitor_logger:
            monitor_logger.stage_start("GENERATING_RESPONSE")

        logger.info(f"Sending to LLM - Prompt: {transcript_text[:100]}...")
        if monitor_logger:
            monitor_logger.debug("LLM", f"Sending prompt to LLM...")

        service = LLMService()
        llm_payload = MessageToLLMRequest(
            prompt=transcript_text,
            system=system,
            history=None,
            temperature=temperature,
        )
        reply = service.message_to_llm(llm_payload)

        if monitor_logger:
            monitor_logger.stage_end("GENERATING_RESPONSE", PipelineStage.DONE)

        logger.info(f"LLM Response: {reply[:100]}...")
        if monitor_logger:
            monitor_logger.debug("LLM", f"Response generated")

        # 6) Save transcription to JSON storage
        transcription_result = _transcription_service.save_transcription(
            user_id=user_id,
            audio_path=str(save_path),
            transcript_text=transcript_text,
            llm_response=reply,
            intent=None,  # Could be populated from brain classification
            temperature=temperature,
        )

        if transcription_result['success']:
            logger.info(f"Transcription saved to: {transcription_result['file_path']}")
            if monitor_logger:
                monitor_logger.debug("API", "Transcription saved")
        else:
            logger.warning(f"Failed to save transcription: {transcription_result['error']}")

        # 7) resposta
        if monitor_logger:
            total_latency = (time.time() - start_time) * 1000
            monitor_logger.record_total_latency(total_latency)
            monitor_logger.record_success()
            monitor_logger.info("API", f"Completed in {total_latency:.1f}ms")

        return UserMessageOut(
            user_id=user_id,
            user_text=transcript_text,
            llm_response=reply,
        )

    except HTTPException:
        if monitor_logger:
            monitor_logger.record_error()
        raise
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}", exc_info=True)
        if monitor_logger:
            monitor_logger.error("API", "Error processing audio", error=str(e))
            monitor_logger.record_error()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/transcriptions/{user_id}")
async def get_user_transcriptions(user_id: str, limit: int = 10):
    """Get transcriptions for a specific user."""
    try:
        transcriptions = _transcription_service.get_user_transcriptions(user_id, limit=limit)
        return {
            "user_id": user_id,
            "count": len(transcriptions),
            "transcriptions": transcriptions,
        }
    except Exception as e:
        logger.error(f"Error retrieving transcriptions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/transcriptions")
async def get_all_transcriptions(limit: int = 50):
    """Get all transcriptions from all users."""
    try:
        transcriptions = _transcription_service.get_all_transcriptions(limit=limit)
        return {
            "total": len(transcriptions),
            "limit": limit,
            "transcriptions": transcriptions,
        }
    except Exception as e:
        logger.error(f"Error retrieving all transcriptions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/transcriptions-stats")
async def get_transcription_statistics(user_id: Optional[str] = None):
    """Get statistics about transcriptions."""
    try:
        stats = _transcription_service.get_statistics(user_id=user_id)
        return stats
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))