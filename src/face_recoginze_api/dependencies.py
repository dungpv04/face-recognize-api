from fastapi import FastAPI, Depends
from services.face_recognize_service import FaceRecognizeService
def get_face_recognize_service(app: FastAPI = Depends()) -> FaceRecognizeService:
    return app.state.faceRecognizeService