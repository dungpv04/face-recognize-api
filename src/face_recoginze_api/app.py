from fastapi import FastAPI, Depends
from face_recoginze_api.routers import faces
from fastapi.middleware.cors import CORSMiddleware


origins = [
    "http://localhost",
    "http://localhost:3000",  # Cho phép React/Vue chạy trên port 3000
    "https://yourdomain.com",
    "*",  # Cho phép tất cả nguồn gốc (KHÔNG NÊN dùng trong production)
]

app = FastAPI()
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

@app.get("/")
async def root():
    return {"message": "Hello Bigger Applications!"}