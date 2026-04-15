from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
from pathlib import Path
from uuid import uuid4
import logging

from orlock.schemas.user_message import UserMessageOut
from orlock.schemas.llm import MessageToLLMRequest
from orlock.services.llm_service import LLMService

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

# Initialize Whisper model (lazy load on first use)
_whisper_model = None

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
        segments = list(model.transcribe(str(audio_path), language="en"))

        # Combine all segments
        transcript = " ".join([
            segment.text.strip() if hasattr(segment, 'text') else str(segment).strip()
            for segment in segments
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
    try:
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

        # 4) TRANSCRIÇÃO - Use actual Whisper transcription
        transcript_text = transcribe_audio(save_path)

        # 5) enviar para LLM
        logger.info(f"Sending to LLM - Prompt: {transcript_text[:100]}...")
        service = LLMService()
        llm_payload = MessageToLLMRequest(
            prompt=transcript_text,
            system=system,
            history=None,
            temperature=temperature,
        )
        reply = service.message_to_llm(llm_payload)
        logger.info(f"LLM Response: {reply[:100]}...")

        # 6) resposta
        return UserMessageOut(
            user_id=user_id,
            user_text=transcript_text,
            llm_response=reply,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))