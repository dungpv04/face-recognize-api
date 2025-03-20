from face_recoginze_api.services.face_recognize_service import FaceRecognizeService, ErrorType
from fastapi import UploadFile, Depends, HTTPException, APIRouter, FastAPI, File, Form
from contextlib import asynccontextmanager
from typing import Annotated
from face_recoginze_api.database.database import Database
from face_recoginze_api.models.response_message import ResponseMessage, STATUS, ResponseSuccesss
from face_recoginze_api.models.face_model import FaceModel
from typing import List
import json
from sqlalchemy.ext.asyncio import AsyncSession
database = Database()

ALLOWED_IMAGE_TYPES = {
    "image/jpeg", "image/png", "image/gif", "image/bmp",
    "image/webp", "image/jfif", "image/avif"
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global faceRecognizeService, database

    async with database.get_session() as db_session:  # ✅ Dùng async with để lấy AsyncSession
        await database.create_db_and_tables()
        faceRecognizeService = FaceRecognizeService()
        await faceRecognizeService.initialize(db_session)
        yield  # Đợi FastAPI chạy app
        faceRecognizeService = None  # Cleanup khi app shutdown


router = APIRouter(lifespan=lifespan)

@router.post("/name", responses={
    200: {
        "description": "✅ Face recognized successfully.",
        "content": {
            "application/json": {
                "example": {
                    "detail":{
                        "status": "SUCCEED",
                        "message": "Found a face that matches the given data",
                        "code": 200,
                        "data": {"username": "john_doe"}
                    }
                }
            }
        }
    },
    400: {
        "description": "⚠️ Invalid request. The uploaded file is missing or incorrect.",
        "content": {
            "application/json": {
                "example": {
                    "detail":{
                        "status": "FAILED",
                        "message": "You must upload exactly 1 file.",
                        "code": 400
                    }
                }
            }
        }
    },
    404: {
        "description": "❌ Face not found in the database.",
        "content": {
            "application/json": {
                "example": {
                    "detail":{
                        "status": "FAILED",
                        "message": "Face not found",
                        "code": 404
                    }
                }
            }
        }
    }
})
async def recognize_face(image: UploadFile):
    if not image.filename:
        raise HTTPException(status_code=400, detail=ResponseMessage(status=STATUS.FAILED, message="You must upload exactly 1 file.", code=400).model_dump())

    #Kiểm tra định dạng file có phải là JPG không
    if not image.filename.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", "jfif")):
        raise HTTPException(status_code=400, detail=ResponseMessage(status=STATUS.FAILED, message="Only image files (JPG, JPEG, PNG, GIF, BMP, WEBP) are allowed.", code=400).model_dump())

    result = faceRecognizeService.recognize_face(file=image)
    if result == ErrorType.FACE_NOT_FOUND.value:
        raise HTTPException(status_code=404, detail=ResponseMessage(status=STATUS.FAILED, message=result, code=404).model_dump())
    elif result == ErrorType.NO_FACE_DETECED.value or result == ErrorType.NOT_MOVING_FACE.value:
        raise HTTPException(status_code=400, detail=ResponseMessage(status=STATUS.FAILED, message=result, code=400).model_dump())
    else:
        return ResponseSuccesss(detail=ResponseMessage(status=STATUS.SUCCEED, message="Found a face matches the given data", code=200, data={'username': result})).model_dump()
    
@router.post("/embedding", responses={
    200: {
        "description": "✅ Face embeddings added to the database successfully.",
        "content": {
            "application/json": {
                "example": {
                    "detail": {
                        "status": "SUCCEED",
                        "message": "Face has been added to the database",
                        "code": 200
                    }
                }
            }
        }
    },
    400: {
        "description": "⚠️ Invalid request. Not enough images uploaded.",
        "content": {
            "application/json": {
                "example": {
                    "detail":{
                        "status": "FAILED",
                        "message": "You must upload at least 5 images",
                        "code": 400
                    }
                }
            }
        }
    },
    409: {
        "description": "❌ Conflict. The face already exists in the database.",
        "content": {
            "application/json": {
                "example": {
                    "detail":{
                        "status": "FAILED",
                        "message": "Face already exists",
                        "code": 409
                    }
                }
            }
        }
    },
    500: {
        "description": "🔥 Internal server error.",
        "content": {
            "application/json": {
                "example": {
                    "detail":{
                        "status": "FAILED",
                        "message": "Internal Server Error",
                        "code": 500
                    }
                }
            }
        }
    }
})
async def add_new_face(data: str = Form(...), images: List[UploadFile] = File(...), session: AsyncSession = Depends(database.get_session)):
    
    try:
        parsed_data = json.loads(data)  # Convert JSON string to dict
        face_data = FaceModel.parse_obj(parsed_data)  # Convert to Pydantic model
        if len(images) < 5:
            raise HTTPException(status_code=400, detail=ResponseMessage(status=STATUS.FAILED, code=400, message="You must upload at least 5 images").model_dump())

        for image in images:
            if not image.filename.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", "jfif", ".avif")) and image.content_type not in ALLOWED_IMAGE_TYPES:
                raise HTTPException(status_code=400, detail=ResponseMessage(status=STATUS.FAILED, message="Only image files (JPG, JPEG, PNG, GIF, BMP, WEBP) are allowed.", code=400).model_dump())

        async with session as db_session:
            result = await faceRecognizeService.generate_face_embeddings(username=face_data.username, files=images, db_session=db_session)
            if result == ErrorType.INTERNAL_SERVER_ERROR.value:
                raise HTTPException(status_code=500)
            elif result == ErrorType.FACE_EXISTED.value or result == ErrorType.USERNAME_EXISTED.value:
                raise HTTPException(status_code=409, detail=ResponseMessage(status=STATUS.FAILED, message=result, code=409).model_dump())
            else:
                return ResponseSuccesss(detail=ResponseMessage(status=STATUS.SUCCEED, message="Face has been added to database", code=200)).model_dump()

    except json.JSONDecodeError:
        return {"error": "Invalid JSON format in 'data'"}


@router.post(
    "/sample",
    summary="Add sample face embeddings to the database",
    description="This endpoint inserts sample face embeddings into the database. It initializes the necessary models and processes the embeddings before storing them.",
    response_description="Returns a success message if the data is added successfully.",
    responses={
        200: {"description": "Sample data added successfully.",
              "content": {
                "application/json": {
                    "example": {
                        "detail":{
                            "status": "SUCCEED",
                            "code": 200,
                            "message": "Sample data added successfully."
                        }
                    }
                }
            }},
        500: {
            "description": "Internal Server Error - Failed to add sample data to the database.",
            "content": {
                "application/json": {
                    "example": {
                        "detail":{
                            "status": "FAILED",
                            "code": 500,
                            "message": "Failed to add sample data to database"
                        }
                    }
                }
            }
        },
    },
)
async def add_sample_data(session: AsyncSession = Depends(database.get_session)):
    """
    Add sample face embeddings to the database.

    This function:
    - Retrieves an active database session.
    - Calls the `generate_face_embeddings_sample` function to add sample data.
    - Returns a success message if successful.
    - Raises an HTTP 500 error if data insertion fails.

    **Returns:**
    - Success message if data is added.
    - HTTP 500 error if data insertion fails.
    """
    async with session as db_session:
        result = await faceRecognizeService.generate_face_embeddings_sample(db_session)
        if result == ErrorType.INTERNAL_SERVER_ERROR.value:
            raise HTTPException(
                status_code=500,
                detail=ResponseMessage(
                    status=STATUS.FAILED,
                    code=500,
                    message="Failed to add sample data to database"
                ).model_dump()
            )
        return result
