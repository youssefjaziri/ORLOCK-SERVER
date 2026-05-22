from fastapi import APIRouter
from orlock.api.v1.endpoints import (
    messagetollm, user_message, user_audio,
    orchestrated_audio, orchestrated_message, orchestrated_intent
)

api_router = APIRouter()

api_router.include_router(messagetollm.router)
api_router.include_router(user_message.router)
api_router.include_router(user_audio.router)

# Orchestrated endpoints (new intelligent layer)
api_router.include_router(orchestrated_audio.router)
api_router.include_router(orchestrated_message.router)
api_router.include_router(orchestrated_intent.router)
