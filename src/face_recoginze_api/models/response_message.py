from pydantic import BaseModel
from typing import Optional
from enum import Enum

class STATUS(Enum):
    SUCCEED = "Succeed"
    FAILED = "Failed"

class ResponseMessage(BaseModel):
    status: str
    message: str
    code: Optional[int] = None
    data: Optional[dict] = None

class ResponseSuccesss(BaseModel):
    detail: ResponseMessage
