from fastapi import APIRouter, HTTPException
from orlock.schemas.llm import MessageToLLMRequest, MessageToLLMResponse
from orlock.services.llm_service import LLMService

router = APIRouter()

@router.post("/messageToLLM", response_model=MessageToLLMResponse)
def message_to_llm(payload: MessageToLLMRequest):
    try:
        service = LLMService()
        text = service.message_to_llm(payload)
        return MessageToLLMResponse(response=text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))