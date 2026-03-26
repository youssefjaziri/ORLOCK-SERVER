from fastapi import APIRouter
from orlock.api.v1.endpoints import messagetollm, user_message, user_audio

api_router = APIRouter()

api_router.include_router(messagetollm.router)
api_router.include_router(user_message.router)
api_router.include_router(user_audio.router)
