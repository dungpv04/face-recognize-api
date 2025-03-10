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
from database.tables import FaceEmbeddingModel
from enum import Enum

class ErrorType(Enum):
    NO_FACE_DETECED = "No face detected"
    FACE_NOT_FOUND = "Face not found"

class FaceRecognizeService:
    def __init__(self, session: Session):
        self.session = session
        
        # Lấy dữ liệu từ DB
        embeddings = self.session.exec(select(FaceEmbeddingModel)).all()
        # Chuyển dữ liệu về NumPy Array
        self.labels = np.array([e.label for e in embeddings])  
        self.vectors = np.array([np.array(e.embedding, dtype=np.float32).mean(axis=0) for e in embeddings])

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
    
    def generate_face_embeddings(self, dataset_path="src/dataset", output_csv="face_embeddings.csv"):
        """
        Quét thư mục dataset, trích xuất embeddings và lưu vào CSV.
        """
        data = []
        
        for root, dirs, files in os.walk(dataset_path):
            label = os.path.basename(root)  # Lấy tên thư mục làm nhãn
            print(f"📂 Đọc thư mục: {label}")

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
                        
                        ypred = self.facenet.embeddings(face_img)
                        data.append([label] + ypred.flatten().tolist())
        
        df = pd.DataFrame(data)
        df.columns = ["label"] + [f"dim_{i}" for i in range(df.shape[1] - 1)]
        df.to_csv(output_csv, index=False)
        
        print("✅ Đã lưu face_embeddings.csv thành công!")
    
    def recognize_face(self, file: UploadFile):
        frame = self.read_image(file)
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # Chuyển BGR → RGB
        results = self.detector.detect_faces(frame_rgb)
        
        if results:
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