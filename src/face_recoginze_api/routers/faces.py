from face_recoginze_api.services.face_recognize_service import FaceRecognizeService, ErrorType
from fastapi import Depends, HTTPException, APIRouter, FastAPI
from contextlib import asynccontextmanager
from typing import Annotated
from face_recoginze_api.database.database import Database
from face_recoginze_api.DTOs.dtos import ResponseMessage, ResponseSuccesss
from face_recoginze_api.DTOs.dtos import ValidateDTO, ImageMetadata
from typing import List
from face_recoginze_api.enums.enums import STATUS
from sqlalchemy.ext.asyncio import AsyncSession
from face_recoginze_api.services.image_service import ReadFileError
from face_recoginze_api.example.example import add_embedding, recognize_face, image_validate
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

@router.get("/name", responses=recognize_face)
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
    return ResponseSuccesss(
        detail=ResponseMessage(
            status=STATUS.SUCCEED, 
            message="Found a face matches the given data", 
            code=200, 
            data=result.model_dump()
            )
        ).model_dump()
    
@router.post("/embedding", responses=add_embedding)
async def add_new_face(
    data: ValidateDTO,
    db: AsyncSession = Depends(database.get_session)):
    async with db as db_session:
        result = await faceRecognizeService.add_new_embedding(data=data, db=db_session)
        if result == ErrorType.IMAGE_HAS_BEEN_USED.value:
                raise HTTPException(
                    status_code=409,
                    detail=ResponseMessage(
                        status=STATUS.FAILED,
                        code=409,
                        message=result
                    ).model_dump()
                )
        elif result == ReadFileError.METADATA_NOT_FOUND.value:
                raise HTTPException(
                    status_code=404,
                    detail=ResponseMessage(
                        status=STATUS.FAILED,
                        code=404,
                        message=result
                    ).model_dump()
                )
        elif result == ErrorType.INTERNAL_SERVER_ERROR.value:
                raise HTTPException(
                    status_code=500,
                    detail=ResponseMessage(
                        status=STATUS.FAILED,
                        code=500,
                        message=result
                    ).model_dump()
                )
        elif result == ErrorType.NO_FACE_DETECED.value or result == ErrorType.IMAGE_NOT_VALIDATE.value or result == ErrorType.USER_FACE_NOT_MATCH.value:
                raise HTTPException(
                    status_code=400,
                    detail=ResponseMessage(
                        status=STATUS.FAILED,
                        code=400,
                        message=result
                    ).model_dump()
                )
        return ResponseSuccesss(
                detail=ResponseMessage(
                    status=STATUS.SUCCEED,
                    code=200,
                    message="Face embedding added successfully!"
                )
            )

@router.post("/validate", responses=image_validate)
async def validate_new_face(img: ImageMetadata, session: AsyncSession = Depends(database.get_session)):
    async with session as db:
        validation_error = await faceRecognizeService.validate_metadata(image_id=img.image_id, db_session=db)
        if validation_error == ErrorType.FACE_EXISTED.value:
            raise HTTPException(
                status_code=409,
                detail=ResponseMessage(
                    status=STATUS.FAILED,
                    code=409,
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
        
        return ResponseSuccesss(
            detail=ResponseMessage(
                status=STATUS.SUCCEED,
                code=200,
                message="Face validated successfully!"
            )
        )
        
        