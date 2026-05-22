"""Orchestrated message endpoint - intelligent text-based response generation."""
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ...schemas.orchestration import OrchestrationResponse
from ...services.orchestration_service import OrchestrationService


logger = logging.getLogger(__name__)
router = APIRouter()

orchestration_service = OrchestrationService()


class UserMessageRequest(BaseModel):
    """Request for orchestrated message processing."""
    user_id: str
    text: str
    system_prompt: str = None


@router.post("/orchestrated/message", response_model=OrchestrationResponse)
async def orchestrated_message(request: UserMessageRequest):
    """
    Orchestrated message endpoint for text-based intelligent response generation.

    Accepts:
    - user_id: User identifier
    - text: User's text input
    - system_prompt: Optional custom system prompt

    Returns:
    - OrchestrationResponse with intent, quality score, and response
    """

    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text input is required")

        logger.debug(f"Processing text message for user {request.user_id}: {request.text[:50]}")

        result = await orchestration_service.process_message(
            user_id=request.user_id,
            user_text=request.text,
            user_system_prompt=request.system_prompt
        )

        logger.info(
            f"Orchestrated text response for {request.user_id}: "
            f"intent={result.intent.value}, "
            f"time={result.processing_time_ms:.0f}ms"
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in orchestrated message endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
