from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
from pathlib import Path
from uuid import uuid4

from orlock.schemas.user_message import UserMessageOut
from orlock.schemas.llm import MessageToLLMRequest
from orlock.services.llm_service import LLMService

router = APIRouter(tags=["audio"])

# pasta onde vais guardar temporariamente
TEMP_AUDIO_DIR = Path("tempaudio")


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
        # se não vier extensão, guarda como .bin (podes trocar para .wav se preferires)
        suffix = original_suffix if original_suffix else ".bin"

        safe_name = f"{user_id}_{uuid4().hex}{suffix}"
        save_path = TEMP_AUDIO_DIR / safe_name

        with open(save_path, "wb") as f:
            f.write(audio_bytes)

        # 4) TRANSCRIÇÃO (placeholder)
        # Troca esta linha pela tua função real de STT/Whisper.
        transcript_text = f"[TRANSCRIPTION_PLACEHOLDER] file saved at {save_path.as_posix()}"

        # 5) enviar para LLM
        service = LLMService()
        llm_payload = MessageToLLMRequest(
            prompt=transcript_text,
            system=system,
            history=None,
            temperature=temperature,
        )
        reply = service.message_to_llm(llm_payload)

        # 6) resposta
        return UserMessageOut(
            user_id=user_id,
            user_text=transcript_text,
            llm_response=reply,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))