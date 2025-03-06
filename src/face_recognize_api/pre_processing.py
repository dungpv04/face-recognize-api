import os
import cv2 as cv
import numpy as np
import pandas as pd
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet

# Ẩn cảnh báo TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Khởi tạo model
detector = MTCNN()
facenet = FaceNet()

# Danh sách để lưu dữ liệu
data = []

folder_path = "src/dataset"

for root, dirs, files in os.walk(folder_path):
    label = os.path.basename(root)  # Lấy tên thư mục làm nhãn
    print(f"📂 Đọc thư mục: {label}")

    for file in files:
        file_path = os.path.join(root, file)
        print(f"  📄 Xử lý: {file_path}")

        # Đọc ảnh
        img_bgr = cv.imread(file_path)
        if img_bgr is None:
            print(f"⚠️ Lỗi đọc ảnh: {file_path}")
            continue

        img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
        results = detector.detect_faces(img_rgb)

        if results:
            x, y, w, h = results[0]['box']

            # Cắt ảnh khuôn mặt
            face_img = img_rgb[y:y+h, x:x+w]
            if face_img.shape[0] > 0 and face_img.shape[1] > 0:
                face_img = cv.resize(face_img, (160, 160))
                face_img = np.expand_dims(face_img, axis=0)

                # Trích xuất embedding
                ypred = facenet.embeddings(face_img)

                # Lưu vào danh sách với nhãn (tên thư mục)
                data.append([label] + ypred.flatten().tolist())

# Chuyển danh sách thành DataFrame
df = pd.DataFrame(data)

# Đặt tên cột
df.columns = ["label"] + [f"dim_{i}" for i in range(df.shape[1] - 1)]

# Lưu vào file CSV
df.to_csv("face_embeddings.csv", index=False)

print("✅ Đã lưu face_embeddings.csv thành công!")
