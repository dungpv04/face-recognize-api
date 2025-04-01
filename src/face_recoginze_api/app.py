from fastapi import FastAPI, Depends
from face_recoginze_api.routers import images, faces, users
from fastapi.middleware.cors import CORSMiddleware
from face_recoginze_api.database.database import Database
from contextlib import asynccontextmanager
from typing import Annotated, AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
origins = [
    "http://localhost",
    "http://localhost:3000",  # Cho phép React/Vue chạy trên port 3000
    "https://yourdomain.com",
    "*",  # Cho phép tất cả nguồn gốc (KHÔNG NÊN dùng trong production)
]

database = Database()
@asynccontextmanager
async def lifespan(app: FastAPI):
    global database
    async with database.get_session() as db_session:  # ✅ Dùng async with để lấy AsyncSession
        await database.create_db_and_tables()
        yield  # Đợi FastAPI chạy app
        database = None


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Danh sách nguồn gốc được phép
    allow_credentials=True,  # Cho phép gửi cookies, xác thực
    allow_methods=["*"],  # Cho phép tất cả phương thức HTTP (GET, POST, PUT, DELETE,...)
    allow_headers=["*"],  # Cho phép tất cả headers
)

app.include_router(
    faces.router,
    prefix="/face",
    tags=["face"],
)

app.include_router(
    users.router,
    prefix="/user",
    tags=["user"],
)


app.include_router(
    images.router,
    prefix="/image", 
    tags=["image"]
)


@app.get("/")
async def root():
    return {"message": "Hello Bigger Applications!"}