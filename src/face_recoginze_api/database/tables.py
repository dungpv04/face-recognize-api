from sqlalchemy.dialects.postgresql import ARRAY
from fastapi import Depends, FastAPI, HTTPException, Query
from sqlmodel import Field, Session, SQLModel, create_engine, select
from sqlalchemy import Column, JSON, ForeignKey
from pgvector.sqlalchemy import Vector

class FaceEmbeddingModel(SQLModel, table=True):
    __tablename__ = "face_embeddings"

    id: int = Field(primary_key=True)
    label: str = Field(nullable=False)
    
class FaceVector(SQLModel, table=True):
    __tablename__ = "face_vector"

    id: int = Field(primary_key=True)
    vector: list[float] = Field(
        sa_column=Column(Vector(512))  # Sử dụng `sa_column` đúng cách
    )
    face_embedding_id: int = Field(sa_column=Column(ForeignKey("face_embeddings.id", ondelete="CASCADE")))
