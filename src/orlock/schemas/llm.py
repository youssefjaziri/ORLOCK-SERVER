from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class MessageToLLMRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    system: Optional[str] = None
    history: Optional[List[ChatMessage]] = None

    # optional generation params
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = None  # some providers ignore; keep for future

class MessageToLLMResponse(BaseModel):
    response: str