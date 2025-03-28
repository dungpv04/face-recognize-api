from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from face_recoginze_api.models.models import Embedding, User
from face_recoginze_api.DTOs.dtos import EmbeddingDTO, UserDTO
from face_recoginze_api.models.models import Image  # Import model Image
from pathlib import Path
from fastapi import UploadFile
from typing import AsyncGenerator
from contextlib import asynccontextmanager
from face_recoginze_api.DTOs.dtos import ImageMetadata
import numpy as np
from sqlmodel import update


#@asynccontextmanager
async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
    async with self.async_session_maker() as session:
        yield session  # ✅ Trả về AsyncSession

async def get_embeddings_with_users(db: AsyncSession) -> list[EmbeddingDTO]:
    query = select(Embedding, User).join(User, Embedding.user_id == User.id)
    results = await db.execute(query)
    rows = results.all()

    return [
        EmbeddingDTO(
            embedding_id=emb.id,
            vector=emb.vector.tolist() if isinstance(emb.vector, np.ndarray) else list(emb.vector),
            user_id=user.id,
            user_name=user.name
        ) for emb, user in rows
    ]

async def save_image_metadata(db: AsyncSession, file: UploadFile, file_path: str) -> ImageMetadata:
    file_size = file.file.seek(0, 2)  # Lấy kích thước file (bytes)
    file.file.seek(0)  # Reset lại vị trí file

    image_metadata = Image(
        filename=file.filename,
        content_type=file.content_type,
        file_size=file_size,
        storage_path=str(Path(file_path).as_posix()),  # Chuẩn hóa đường dẫn với `/`
    )

    db.add(image_metadata)
    await db.commit()
    await db.refresh(image_metadata)
    result = ImageMetadata(image_id=image_metadata.id)
    return result  # Trả về metadata sau khi lưu

async def get_storage_path_by_id(session: AsyncSession, image_id: int):
    statement = select(Image.storage_path).where(Image.id == image_id)
    result = await session.execute(statement)
    metadata = result.first()
    if metadata:
        return metadata[0]
    return None  # Trả về storage_path hoặc None nếu không tìm thấy

async def get_username_by_id(db: AsyncSession, user_id: int) -> str | None:
    query = select(User.name).where(User.id == user_id)
    result = await db.execute(query)
    username = result.scalar_one_or_none()
    return username


async def get_user_by_dto(dto: UserDTO, session: AsyncSession) -> int | None:
    statement = select(User.id).where(User.name == dto.username)
    result = await session.execute(statement)  # Chờ kết quả từ DB
    user_id = result.scalar_one_or_none()  # Lấy giá trị của cột đầu tiên hoặc None
    return user_id
async def update_is_validate_by_image_id(image_id: int, db: AsyncSession) -> bool:
    statement = (
        update(Image)
        .where(Image.id == image_id)
        .values(is_validate=True)
    )
    result = await db.execute(statement)
    await db.commit()  # Lưu thay đổi vào DB

    return result.rowcount > 0  # Trả về True nếu có ít nhất 1 dòng được cập nhật
