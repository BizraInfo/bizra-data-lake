from app.routers import router
from fastapi import FastAPI

APP_VERSION = "4.0.1"

app = FastAPI(
    title="BIZRA URP Knowledge Graph",
    version=APP_VERSION,
)
app.include_router(router)
