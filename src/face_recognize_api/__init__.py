from face_recognize_api.FaceRecognizeService import FaceRecognizeService
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}
def main() -> None:
    faceRecognizeService = FaceRecognizeService()
    faceRecognizeService.start_camera()
