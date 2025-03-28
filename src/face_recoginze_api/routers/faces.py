from face_recoginze_api.services.face_recognize_service import FaceRecognizeService, ErrorType
from fastapi import UploadFile, Depends, HTTPException, APIRouter, FastAPI, File, Form
from contextlib import asynccontextmanager
from typing import Annotated
from face_recoginze_api.database.database import Database
from face_recoginze_api.DTOs.dtos import ResponseMessage, STATUS, ResponseSuccesss
from face_recoginze_api.DTOs.dtos import ValidateDTO
from typing import List
import json
from sqlalchemy.ext.asyncio import AsyncSession
from face_recoginze_api.services.image_service import ReadFileError
database = Database()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global faceRecognizeService, database
    print("Khởi tạo lifespan trong faces")
    async with database.get_session() as db_session:  # ✅ Dùng async with để lấy AsyncSession
        faceRecognizeService = FaceRecognizeService()
        await faceRecognizeService.init_faiss_index(db_session=db_session)
        yield  # Đợi FastAPI chạy app
        faceRecognizeService = None  # Cleanup khi app shutdown

router = APIRouter(lifespan=lifespan)

@router.get("/name", responses={
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
async def recognize_face(image_id: int, session: AsyncSession = Depends(database.get_session)):
    async with session as db_session:
        error, result = await faceRecognizeService.recognize_face_faiss(image_id=image_id, db=db_session)
    if error == ErrorType.FACE_NOT_FOUND.value:
        raise HTTPException(
            status_code=404, 
            detail=ResponseMessage(
                status=STATUS.FAILED, 
                message=error, 
                code=404
                ).model_dump()
            )
    elif error == ErrorType.NO_FACE_DETECED.value or error == ErrorType.NOT_MOVING_FACE.value:
        raise HTTPException(
            status_code=400, 
            detail=ResponseMessage(
                status=STATUS.FAILED, 
                message=error, 
                code=400
                ).model_dump()
            )
    elif error == ErrorType.INTERNAL_SERVER_ERROR.value:
        raise HTTPException(
            status_code=500, 
            detail=ResponseMessage(
                status=STATUS.FAILED, 
                message=error, 
                code=400
                ).model_dump()
            )
    elif error == ReadFileError.FILE_NOT_FOUND.value or error == ReadFileError.METADATA_NOT_FOUND.value:
        raise HTTPException(
            status_code=404,
            detail=ResponseMessage(
                status=STATUS.FAILED,
                message=error,
                code=404
            ).model_dump()
        )
    else:
        return ResponseSuccesss(
            detail=ResponseMessage(
                status=STATUS.SUCCEED, 
                message="Found a face matches the given data", 
                code=200, 
                data=result.model_dump()
                )
            ).model_dump()
    
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
        face_data = UserDTO.parse_obj(parsed_data)  # Convert to Pydantic model
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

@router.post("/validate")
async def validate_new_face(validate: ValidateDTO, session: AsyncSession = Depends(database.get_session)):
    async with session as db:
        validation_error = await faceRecognizeService.validate_add_request(validateDTO=validate, db_session=db)
        if validation_error == ErrorType.USER_EXISTED.value or validation_error == ErrorType.FACE_EXISTED.value:
            raise HTTPException(
                status_code=409,
                detail=ResponseMessage(
                    status=STATUS.FAILED,
                    code=409,
                    message=validation_error
                ).model_dump()
            )
        elif validation_error == ErrorType.NO_FACE_DETECED.value:
            raise HTTPException(
                status_code=400,
                detail=ResponseMessage(
                    status=STATUS.FAILED,
                    code=400,
                    message=validation_error
                ).model_dump()
            )
        elif validation_error == ErrorType.INTERNAL_SERVER_ERROR.value:
            raise HTTPException(
                status_code=500,
                detail=ResponseMessage(
                    status=STATUS.FAILED,
                    code=500,
                    message=validation_error
                ).model_dump()
            )
        elif validation_error == ReadFileError.METADATA_NOT_FOUND.value or validation_error == ReadFileError.FILE_NOT_FOUND.value:
            raise HTTPException(
                status_code=404,
                detail=ResponseMessage(
                    status=STATUS.FAILED,
                    code=404,
                    message=validation_error
                ).model_dump()
            )
        else:
            return ResponseSuccesss(
                detail=ResponseMessage(
                    status=STATUS.SUCCEED,
                    code=200,
                    message="User information and face validated successfully!"
                )
            )