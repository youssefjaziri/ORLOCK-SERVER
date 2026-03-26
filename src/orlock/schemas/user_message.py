from pydantic import BaseModel, Field
from typing import Optional

class UserMessageIn(BaseModel):
    user_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    system: Optional[str] = None
    temperature: Optional[float] = 0.2

class UserMessageOut(BaseModel):
    user_id: str
    user_text: str
    llm_response: str