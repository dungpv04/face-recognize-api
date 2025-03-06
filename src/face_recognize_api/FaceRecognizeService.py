import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import faiss
import cv2 as cv
from mtcnn import MTCNN
from keras_facenet import FaceNet

class FaceRecognizeService:
    def __init__(self, csv_path="./src/face_embeddings.csv"):
        # Đọc dữ liệu từ file CSV
        self.df = pd.read_csv(csv_path)
        
        # Lấy nhãn và vector embeddings
        self.labels = self.df["label"].values  # Nhãn (tên thư mục)
        self.vectors = self.df.iloc[:, 1:].values.astype("float32")  # Tất cả cột embedding (float32)
        
        # Ánh xạ chỉ số FAISS -> tên người
        self.index_to_name = {i: name for i, name in enumerate(self.labels)}
        
        # Khởi tạo FAISS Index
        dimension = self.vectors.shape[1]  # Số chiều của vector
        self.index = faiss.IndexFlatL2(dimension)  # Dùng L2 (Euclidean Distance)
        self.index.add(self.vectors)  # Thêm vector vào FAISS
        
        # Khởi tạo MTCNN và FaceNet
        self.detector = MTCNN()
        self.facenet = FaceNet()
        
        # Khởi tạo VideoCapture
        self.capture = cv.VideoCapture(0)
    
    def recognize_face_faiss(self, face_vector, top_k=1, threshold=1.0):
        """
        Tìm người gần nhất với face_vector bằng FAISS.
        Nếu khoảng cách > threshold, trả về 'Unknown'.
        """
        face_vector = np.array(face_vector).astype('float32').reshape(1, -1)
        D, I = self.index.search(face_vector, top_k)  # D: khoảng cách, I: chỉ số
        
        best_index = I[0][0]
        best_distance = D[0][0]
        
        if best_distance > threshold:
            return "Unknown", best_distance
        
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
    
    def start_camera(self):
        """
        Bắt đầu camera để nhận diện khuôn mặt theo thời gian thực.
        """
        if not self.capture.isOpened():
            print("❌ Không thể mở camera")
            exit()

        frame_count = 1

        while True:
            ret, frame = self.capture.read()
            if not ret:
                print("❌ Không thể đọc khung hình")
                break

            frame = cv.flip(frame, 1)  # Lật ngang giống gương
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # Chuyển BGR → RGB
            results = self.detector.detect_faces(frame_rgb)
            
            if results:
                x, y, w, h = results[0]['box']
                cv.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)

                frame_count += 1
                face_img = frame[y: y+h, x: x+w]
                if face_img.shape[0] > 0 and face_img.shape[1] > 0 and frame_count % 10 == 0:
                    face_img = cv.resize(face_img, (160, 160))
                    face_img = np.expand_dims(face_img, axis=0)
                    ypred = self.facenet.embeddings(face_img)
                    frame_count = 1
                    predicted_name, confidence = self.recognize_face_faiss(ypred)
                    print(f"Đối tượng: {predicted_name}")
            
            frame_bgr = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)  # Chuyển lại về BGR trước khi hiển thị
            cv.imshow("Camera", frame_bgr)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        self.capture.release()
        cv.destroyAllWindows()
