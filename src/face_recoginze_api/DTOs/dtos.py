from pydantic import BaseModel
from typing import Optional
from enum import Enum
from typing import List

class STATUS(Enum):
    SUCCEED = "Succeed"
    FAILED = "Failed"

class ResponseMessage(BaseModel):
    status: str
    message: str
    code: Optional[int] = None
    data: Optional[object] = None

class ResponseSuccesss(BaseModel):
    detail: ResponseMessage

class EmbeddingDTO(BaseModel):
    embedding_id: int
    vector: List[float]
    user_id: int
    user_name: str

class UserDTO(BaseModel):
    username: str

class ImageMetadata(BaseModel):
    image_id: int

class ValidateDTO(BaseModel):
    image: ImageMetadata
    user: UserDTO