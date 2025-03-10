from mtcnn import MTCNN
from keras_facenet import FaceNet

def get_mtcnn():
    return MTCNN()

def get_facenet():
    return FaceNet()
