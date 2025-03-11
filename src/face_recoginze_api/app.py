from fastapi import FastAPI, Depends
from routers import faces
from contextlib import asynccontextmanager
from services.face_recognize_service import FaceRecognizeService
from database.database import Database

app = FastAPI()

app.include_router(
    faces.router,
    prefix="/face",
    tags=["face"],
)

@app.get("/")
async def root():
    return {"message": "Hello Bigger Applications!"}