import os
from fastapi import UploadFile
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from typing import List
import pandas as pd
import numpy as np
import faiss
import cv2 as cv
from mtcnn import MTCNN
from keras_facenet import FaceNet
from fastapi import HTTPException
from sqlmodel import Session, select
from sqlmodel.ext.asyncio.session import AsyncSession
from enum import Enum
from face_recoginze_api.repositories.repositories import get_embeddings_with_users, get_username_by_id, get_user_by_dto, update_is_validate_by_image_id
from fastapi import Depends
from face_recoginze_api.services.image_service import ImageService
from typing import Annotated
from face_recoginze_api.DTOs.dtos import UserDTO, ValidateDTO
from face_recoginze_api.enums.enums import ErrorType

class FaceRecognizeService:

    def __init__(self):

        self.image_service = ImageService()
        self.detector = MTCNN()
        self.facenet = FaceNet()
        self.index = None
        self.index_to_name = {}

    async def init_faiss_index(self, db_session: AsyncSession):
        embeddings = await get_embeddings_with_users(db_session)
        if len(embeddings) > 0:
            print("Initialize Faiss Index ...")
            self.labels = np.array([e.user_id for e in embeddings])  
            self.vectors = np.array([np.array(e.vector, dtype=np.float32) for e in embeddings])

            # Khởi tạo FAISS Index
            dimension = self.vectors.shape[1]
            self.index = faiss.IndexHNSWFlat(dimension, 32)
            self.index.add(self.vectors)

            # Ánh xạ chỉ số FAISS -> tên người
            self.index_to_name = {i: name for i, name in enumerate(self.labels)}

    
    async def recognize_face_faiss(self, db: AsyncSession, image_id, top_k=5, threshold=1.0) -> tuple[str, UserDTO]:
        """
        Tìm người gần nhất với face_vector bằng FAISS.
        Nếu khoảng cách > threshold, trả về 'Unknown'.
        """
        error, face_vector = await self.generate_face_embedding_from_image(db=db, image_id=image_id)

        if error:
            return error, None

        face_vector = np.array(face_vector).astype('float32').reshape(1, -1)
        D, I = self.index.search(face_vector, top_k)
        
        best_index = I[0][0]
        best_distance = D[0][0]
        
        if best_distance > threshold:
            return ErrorType.FACE_NOT_FOUND.value, None
        
        predicted_user_id = int(self.index_to_name[best_index])
        username = await get_username_by_id(db = db, user_id=predicted_user_id)
        return None, UserDTO(username=username)
    
    async def generate_face_embedding_from_image(self, image_id: int, db: AsyncSession):
        error, img_content = await self.image_service.read_img_by_id(image_id=image_id, db=db)
        if error:
            return error, None

        try:
            # Chuyển đổi bytes thành numpy array trước khi decode
            np_img = np.frombuffer(img_content, np.uint8)
            frame = cv.imdecode(np_img, cv.IMREAD_COLOR)

            if frame is None:
                print("Error: Image decoding failed.")
                return ErrorType.INTERNAL_SERVER_ERROR.value, None  # 🔥 Trả về None nếu ảnh không hợp lệ

            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = self.detector.detect_faces(frame_rgb)
        except Exception as e:
            print(f"Error processing image: {e}")
            return ErrorType.INTERNAL_SERVER_ERROR.value
            
        if results:
            print(results)
            x, y, w, h = results[0]['box']
            face_img = frame[y: y+h, x: x+w]
            face_img = cv.resize(face_img, (160, 160))
            face_img = np.expand_dims(face_img, axis=0)
            embedding = self.facenet.embeddings(face_img)
            return None, embedding
        return ErrorType.NO_FACE_DETECED.value, None

    async def validate_face(self, image_id: int, db_session: AsyncSession) -> str:
        error, is_face_exist = await self.recognize_face_faiss(db=db_session, image_id=image_id)
        if error is None or error:
            return ErrorType.FACE_EXISTED.value
        elif error == ErrorType.FACE_NOT_FOUND.value:
            return None
        else:
            return error
    
    async def validate_user_data(self, userDTO: UserDTO, db_session: AsyncSession) -> str:
        is_user_exist = await get_user_by_dto(dto=userDTO, session=db_session)
        if is_user_exist:
            return ErrorType.USER_EXISTED.value
        return None
    
    async def validate_add_request(self, validateDTO: ValidateDTO, db_session: AsyncSession):
        userDTO = validateDTO.user
        imageDTO = validateDTO.image
        face_error = await self.validate_face(image_id=imageDTO.image_id, db_session=db_session) 
        user_data_error = await self.validate_user_data(userDTO=userDTO, db_session=db_session)
        if face_error:
            await self.image_service.delete_img_by_id(image_id=imageDTO.image_id, db=db_session)
            return face_error
        if user_data_error:
            return user_data_error
        row_updated = await update_is_validate_by_image_id(image_id=imageDTO.image_id, db=db_session)
        if row_updated:
            return None
        return ErrorType.INTERNAL_SERVER_ERROR.value