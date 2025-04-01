from fastapi import FastAPI, APIRouter, HTTPException, Depends
from typing import Dict, Annotated
from face_recoginze_api.services.user_services import UserService
from face_recoginze_api.DTOs.dtos import UserDTO
from sqlalchemy.ext.asyncio import AsyncSession
from face_recoginze_api.database.database import Database
from face_recoginze_api.DTOs.dtos import ResponseSuccesss, ResponseMessage
from face_recoginze_api.enums.enums import STATUS
from face_recoginze_api.example.example import add_user, delete_user
router = APIRouter()
database = Database()

@router.post("/", responses=add_user)
async def create_user(userDTO: UserDTO, 
                session: AsyncSession = Depends(database.get_session)):
    async with session as db:
        service = UserService(session=db)
        new_user_id = await service.create_user(userDTO)
        return new_user_id
    return ResponseMessage(
                status=STATUS.FAILED,
                message="Internal Server error.",
                code=500
        )

@router.delete("/{user_id}", responses=delete_user)
async def delete_user(user_id: int, session: AsyncSession = Depends(database.get_session)):
    async with session as db:
        service = UserService(session=db)
        delete = await service.delete_user(user_id=user_id)
        if delete:
            return ResponseSuccesss(
                detail=ResponseMessage(
                    status=STATUS.SUCCEED,
                    message="Deleted user successfully.",
                    code=200
                )
            )
        return ResponseMessage(
                status=STATUS.FAILED,
                message="Internal Server error.",
                code=500
        )
