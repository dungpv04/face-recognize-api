from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlmodel import Field, Session, SQLModel, create_engine, select, ForeignKey
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class Image(SQLModel, table=True):
    __tablename__ = "images"

    id: int = Field(primary_key=True, index=True)
    filename: str = Field(nullable=False)  # Tên file
    content_type: str = Field(nullable=False)  # Loại MIME (image/png, video/mp4, etc.)
    file_size: int | None = Field(default=None)  # Kích thước file (bytes), có thể null
    storage_path: str = Field(nullable=False)  # Đường dẫn lưu file
    is_validate: bool = Field(nullable=False, default=False)
class User(SQLModel, table=True):
    __tablename__ = "users"

    id: int = Field(primary_key=True)
    name: str = Field(nullable=False)

class Embedding(SQLModel, table=True):
    __tablename__ = "embeddings"

    id: int = Field(primary_key=True)
    vector: list[float] = Field(
        sa_column=Column(Vector(512))
    )
    user_id: int = Field(sa_column=Column(ForeignKey("users.id", ondelete="CASCADE")))
    image_id: int = Field(sa_column=Column(ForeignKey("images.id", ondelete="CASCADE")))

