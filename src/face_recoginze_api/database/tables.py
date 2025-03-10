from sqlalchemy.dialects.postgresql import ARRAY
from fastapi import Depends, FastAPI, HTTPException, Query
from sqlmodel import Field, Session, SQLModel, create_engine, select
from sqlalchemy import Column, JSON

class FaceEmbeddingModel(SQLModel, table=True):
    __tablename__ = "face_embeddings"

    id: int = Field(primary_key=True)
    label: str = Field(nullable=False)
    embedding: list[list[float]] | None = Field(default=None, sa_column=Column(JSON))
