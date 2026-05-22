"""Intent detection endpoint for debugging."""
import logging
from fastapi import APIRouter, HTTPException
from ...schemas.intent import IntentResult
from ...intent.detector import IntentDetector


logger = logging.getLogger(__name__)
router = APIRouter()

intent_detector = IntentDetector()


@router.post("/debug/intent", response_model=IntentResult)
async def detect_intent(text: str):
    """
    Debug endpoint for intent detection.

    Accepts:
    - text: User input text to classify

    Returns:
    - IntentResult with category, confidence, and reasoning
    """

    try:
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Text input is required")

        logger.debug(f"Detecting intent for: {text[:50]}")

        result = await intent_detector.detect(text)

        logger.debug(
            f"Intent detected: {result.category.value} "
            f"(confidence: {result.confidence:.2f})"
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in intent detection endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Intent detection failed: {str(e)}")
