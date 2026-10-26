from fastapi import FastAPI
from src.api.endpoints import router
from src.core.config import settings

app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)

app.include_router(router)
