import os
from fastapi import UploadFile
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import faiss
import cv2 as cv
from mtcnn import MTCNN
from keras_facenet import FaceNet
from fastapi import HTTPException
from sqlmodel import Session, select
from face_recoginze_api.database.tables import FaceEmbeddingModel, FaceVector
from sqlmodel.ext.asyncio.session import AsyncSession
from enum import Enum
import tempfile

class ErrorType(Enum):
    NO_FACE_DETECED = "No face detected"
    FACE_NOT_FOUND = "Face not found"
    NOT_MOVING_FACE = "Not a moving face"

class FaceRecognizeService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def initialize(self):
        # Truy vấn dữ liệu từ cả hai bảng bằng JOIN
        result = await self.session.execute(
            select(FaceEmbeddingModel.label, FaceVector.vector)
            .join(FaceVector, FaceVector.face_embedding_id == FaceEmbeddingModel.id)
        )
        embeddings = result.all()  # Lấy tất cả kết quả

        # Chuyển dữ liệu về NumPy Array
        self.labels = np.array([e[0] for e in embeddings])  
        self.vectors = np.array([np.array(e[1], dtype=np.float32) for e in embeddings])

        # Ánh xạ chỉ số FAISS -> tên người
        self.index_to_name = {i: name for i, name in enumerate(self.labels)}

        # Khởi tạo FAISS Index
        dimension = self.vectors.shape[1]
        self.index = faiss.IndexHNSWFlat(dimension, 32)  # Faster Approximate Search
        self.index.add(self.vectors)

        self.detector = MTCNN()
        self.facenet = FaceNet()
    
    def recognize_face_faiss(self, face_vector, top_k=4, threshold=1.0):
        """
        Tìm người gần nhất với face_vector bằng FAISS.
        Nếu khoảng cách > threshold, trả về 'Unknown'.
        """
        face_vector = np.array(face_vector).astype('float32').reshape(1, -1)
        D, I = self.index.search(face_vector, top_k)
        
        best_index = I[0][0]
        best_distance = D[0][0]
        
        if best_distance > threshold:
            return None, None
        
        return self.index_to_name[best_index], best_distance
    
    async def generate_face_embeddings_sample(self, dataset_path="../dataset", db_session: AsyncSession = None):
        if db_session is None:
            raise ValueError("⚠️ Cần cung cấp db_session để kết nối database!")

        num_folders = 0
        num_files = 0

        for root, dirs, files in os.walk(dataset_path):
            num_folders += len(dirs)
            num_files += len(files)

        print(f"{num_files} and {num_folders}")

        async with db_session.begin():  
            for root, dirs, files in os.walk(dataset_path):
                label = os.path.basename(root)
                print(f"📂 Đọc thư mục: {label}")

                # Kiểm tra xem người dùng đã tồn tại chưa
                statement = select(FaceEmbeddingModel).where(FaceEmbeddingModel.label == label)
                result = await db_session.execute(statement)
                face = result.scalars().first()

                if face:
                    face_embedding_id = face.id
                else:
                    new_face = FaceEmbeddingModel(label=label)
                    db_session.add(new_face)
                    await db_session.flush()  
                    face_embedding_id = new_face.id

                for file in files:
                    file_path = os.path.join(root, file)
                    print(f"  📄 Xử lý: {file_path}")

                    img_bgr = cv.imread(file_path)
                    if img_bgr is None:
                        print(f"⚠️ Lỗi đọc ảnh: {file_path}")
                        continue

                    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
                    results = self.detector.detect_faces(img_rgb)

                    if results:
                        x, y, w, h = results[0]['box']
                        face_img = img_rgb[y:y+h, x:x+w]

                        if face_img.shape[0] > 0 and face_img.shape[1] > 0:
                            face_img = cv.resize(face_img, (160, 160))
                            face_img = np.expand_dims(face_img, axis=0)

                            # Lấy embeddings
                            ypred = self.facenet.embeddings(face_img).flatten().tolist()
                            print(f"🎯 Embedding tạo thành công: {ypred[:5]}...")  # Debug

                            # Lưu vào DB
                            new_vector = FaceVector(vector=ypred, face_embedding_id=face_embedding_id)
                            db_session.add(new_vector)
                            print(f"📝 Đã thêm vector vào DB: {new_vector}")

            await db_session.commit()  
            print("✅ Đã commit dữ liệu vào PostgreSQL thành công!")

    
    def recognize_face(self, file: UploadFile):
        # if not self.validate_face(file):
        #     return ErrorType.NOT_MOVING_FACE.value

        frame = self.read_image(file)
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # Chuyển BGR → RGB
        results = self.detector.detect_faces(frame_rgb)
        
        if results:
            print(results)
            x, y, w, h = results[0]['box']
            face_img = frame[y: y+h, x: x+w]
            face_img = cv.resize(face_img, (160, 160))
            face_img = np.expand_dims(face_img, axis=0)
            embedding = self.facenet.embeddings(face_img)
            predicted_name, confidence = self.recognize_face_faiss(embedding)
            if predicted_name:
                return predicted_name
            return ErrorType.FACE_NOT_FOUND.value
        return ErrorType.NO_FACE_DETECED.value
        

    def read_image(self, file: UploadFile):
        
        file_bytes = np.frombuffer(file.file.read(), np.uint8)
        frame = cv.imdecode(file_bytes, cv.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=500, detail="An error occur while trying to read image.")

        return frame
    
    def read_video(self, file: UploadFile):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(file.file.read())
            return temp_video.name
    
    def validate_face(self, file: UploadFile):
        vertical_move = False
        horizontal_move = False
        cap = cv.VideoCapture(self.read_video(file))
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv.flip(frame, 1)
            if not ret:
                break

            # Chuyển ảnh sang RGB (MTCNN yêu cầu định dạng này)
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            # Phát hiện khuôn mặt
            faces = self.detector.detect_faces(rgb_frame)
            prev_keypoints = None
            if faces:
                for face in faces:
                    keypoints = face["keypoints"]

                    # Lấy vị trí mắt và mũi
                    left_eye = np.array(keypoints["left_eye"])
                    right_eye = np.array(keypoints["right_eye"])
                    nose = np.array(keypoints["nose"])

                    # Vẽ keypoints lên ảnh
                    x, y, width, height = face["box"]
                    cv.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

                    # Nếu đã có frame trước, so sánh vị trí để xác định hướng di chuyển
                    if prev_keypoints is not None:
                        prev_nose = prev_keypoints["nose"]

                        movement = nose - prev_nose
                        direction = ""

                        if movement[0] > 5:
                            direction = "Right →"
                            horizontal_move = True
                        elif movement[0] < -5:
                            direction = "← Left"
                            horizontal_move = True

                        if movement[1] > 5:
                            direction += " ↓ Down"
                            vertical_move = True
                        elif movement[1] < -5:
                            direction += " ↑ Up"
                            vertical_move = True

                        # Hiển thị hướng di chuyển lên màn hình
                        print(f"Direction: {direction}")

                        left_eye = keypoints['left_eye']
                        right_eye = keypoints['right_eye']

                    # Cập nhật keypoints của frame trước
                    prev_keypoints = keypoints
        return vertical_move and horizontal_move
