import cv2
import numpy as np
from mtcnn import MTCNN

# Khởi tạo MTCNN detector
detector = MTCNN()

# Khởi động webcam
cap = cv2.VideoCapture(0)

# Lưu vị trí keypoints của frame trước
prev_keypoints = None




# Giải phóng bộ nhớ
cap.release()
cv2.destroyAllWindows()
