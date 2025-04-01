from sqlmodel.ext.asyncio.session import AsyncSession
from typing import Optional, List
from face_recoginze_api.models.models import User
from face_recoginze_api.repositories.repositories import (
    statement_get_user_by_id,
    statement_get_all_users,
    statement_create_user,
    statement_delete_user_by_id,
)
from face_recoginze_api.DTOs.dtos import UserDTO

class UserService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_user(self, userDTO: UserDTO) -> Optional[int]:
        """Tạo user mới"""
        new_user = statement_create_user(userDTO=userDTO)
        self.session.add(new_user)
        await self.session.commit()
        await self.session.refresh(new_user)
        return new_user.id

    async def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Lấy user theo ID"""
        statement = statement_get_user_by_id(user_id)
        result = await self.session.exec(statement)
        return result.first()

    async def get_all_users(self) -> List[User]:
        """Lấy danh sách tất cả users"""
        statement = statement_get_all_users()
        result = await self.session.exec(statement)
        return result.all()

    async def update_user_name(self, user_id: int, new_name: str) -> bool:
        """Cập nhật tên user"""
        user = await self.get_user_by_id(user_id)
        if user:
            user.name = new_name
            self.session.add(user)
            await self.session.commit()
            await self.session.refresh(user)
            return True
        return False

    async def delete_user(self, user_id: int) -> bool:
        """Xóa user theo ID"""
        statement = statement_delete_user_by_id(user_id)
        result = await self.session.execute(statement)
        await self.session.commit()
        return result.rowcount > 0
