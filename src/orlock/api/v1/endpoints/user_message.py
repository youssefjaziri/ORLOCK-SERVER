from fastapi import APIRouter, HTTPException
from orlock.schemas.user_message import UserMessageIn, UserMessageOut
from orlock.schemas.llm import MessageToLLMRequest
from orlock.services.llm_service import LLMService

router = APIRouter(tags=["messages"])

@router.post("/userMessage", response_model=UserMessageOut)
def user_message(payload: UserMessageIn):
    try:
        service = LLMService()

        llm_payload = MessageToLLMRequest(
            prompt=payload.text,
            system=payload.system,
            history=None,
            temperature=payload.temperature
        )

        reply = service.message_to_llm(llm_payload)

        return UserMessageOut(
            user_id=payload.user_id,
            user_text=payload.text,
            llm_response=reply
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))