from fastapi import FastAPI
from orlock.api.v1.router import api_router

app = FastAPI(title="ORLOCK Backend")
app.include_router(api_router, prefix="/api/v1")