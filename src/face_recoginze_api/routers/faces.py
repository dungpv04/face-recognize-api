from services.face_recognize_service import FaceRecognizeService, ErrorType
from fastapi import UploadFile, Depends, HTTPException, APIRouter, FastAPI
from contextlib import asynccontextmanager
from typing import Annotated
from sqlmodel import Session
from database.database import Database
from models.response_message import ResponseMessage, STATUS

database = Database()
database.create_db_and_tables()
SessionDep = Annotated[Session, Depends(database.get_session)]

@asynccontextmanager
async def lifespan(app: FastAPI):
    global faceRecognizeService, database, SessionDep
    with database.get_session() as session:  # ✅ Sửa next() thành with
        faceRecognizeService = FaceRecognizeService(session)
        yield  # Đợi FastAPI chạy app
        faceRecognizeService = None  # Cleanup khi app shutdown

router = APIRouter(lifespan=lifespan)

@router.post("/", responses={
    404: {
        "description": 'Not Found Exception',
        "content": {
                "application/json": {
                    "example": {"detail": "Face not found"}
                }
            }
        }
    }
)
async def recognize_face(image: UploadFile):
    if not image.filename:
        raise HTTPException(status_code=400, detail=ResponseMessage(status=STATUS.FAILED, message="You must upload exactly 1 file.", code=400).model_dump())

    # Kiểm tra định dạng file có phải là JPG không
    # if not image.filename.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", "jfif")):
    #     raise HTTPException(status_code=400, detail=ResponseMessage(status=STATUS.FAILED, message="Only image files (JPG, JPEG, PNG, GIF, BMP, WEBP) are allowed.", code=400).model_dump())

    result = faceRecognizeService.recognize_face(file=image)
    if result == ErrorType.FACE_NOT_FOUND.value or result == ErrorType.NO_FACE_DETECED.value or result == ErrorType.NOT_MOVING_FACE.value:
        raise HTTPException(status_code=404, detail=ResponseMessage(status=STATUS.FAILED, message=result, code=404).model_dump())
    else:
        return ResponseMessage(status=STATUS.SUCCEED, message="Found a face matches the given data", code=200, data={'username': result})
    
    
