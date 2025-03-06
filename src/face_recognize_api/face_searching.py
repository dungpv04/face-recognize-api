import faiss
import numpy as np
import pandas as pd

# ==== Bước 1: Đọc dữ liệu từ CSV ====
df = pd.read_csv("./src/face_embeddings.csv")

# Lấy nhãn và vector embeddings
labels = df["label"].values  # Nhãn (tên thư mục)
vectors = df.iloc[:, 1:].values.astype("float32")  # Lấy tất cả cột embedding (float32)

# Ánh xạ chỉ số FAISS -> tên người
index_to_name = {i: name for i, name in enumerate(labels)}

# ==== Bước 2: Khởi tạo FAISS Index ====
dimension = vectors.shape[1]  # Số chiều của vector
index = faiss.IndexFlatL2(dimension)  # Dùng L2 (Euclidean Distance)
index.add(vectors)  # Thêm vector vào FAISS

# ==== Bước 3: Tìm kiếm vector mới ====
def recognize_face_faiss(face_vector, top_k=1, threshold=1.0):
    """
    Tìm người gần nhất với face_vector bằng FAISS.
    Nếu khoảng cách > threshold, trả về 'Unknown'.
    """
    face_vector = np.array(face_vector).astype('float32').reshape(1, -1)
    D, I = index.search(face_vector, top_k)  # D: khoảng cách, I: chỉ số

    best_index = I[0][0]
    best_distance = D[0][0]

    if best_distance > threshold:
        return "Unknown", best_distance

    return index_to_name[best_index], best_distance
