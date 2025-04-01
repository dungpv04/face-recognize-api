from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from face_recoginze_api.services.image_service import ImageService
from face_recoginze_api.DTOs.dtos import ResponseMessage, ResponseSuccesss
from face_recoginze_api.database.database import Database
from typing import Annotated
from face_recoginze_api.enums.enums import STATUS
from face_recoginze_api.example.example import upload_image
router = APIRouter()

ALLOWED_IMAGE_TYPES = {
    "image/jpeg", "image/png", "image/gif", "image/bmp",
    "image/webp"
}

@router.post("/image/", responses=upload_image)
async def upload_image(image_service: Annotated[ImageService, Depends(ImageService)], 
                       db: Annotated[Database, Depends(Database)],
                       file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400, 
            detail=ResponseMessage(
                status=STATUS.FAILED, 
                message="Only image files (JPG, JPEG, PNG, GIF, BMP, WEBP) are allowed.", 
                code=400
                ).model_dump()
            )

    try:
        async with db.get_session() as db_session:
            result = await image_service.save_image(file, db_session)
            return ResponseSuccesss(
                detail=ResponseMessage(
                    status=STATUS.SUCCEED,
                    code=200,
                    data=result.model_dump(),
                    message="Upload image successfully!"
                ).model_dump()
            )
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, 
                            detail=ResponseMessage(
                                status=STATUS.FAILED,
                                message="Internal server error!",
                                code=500
                            ).model_dump())
