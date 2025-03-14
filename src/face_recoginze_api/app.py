from fastapi import FastAPI, Depends
from face_recoginze_api.routers import faces

app = FastAPI()

app.include_router(
    faces.router,
    prefix="/face",
    tags=["face"],
)

@app.get("/")
async def root():
    return {"message": "Hello Bigger Applications!"}