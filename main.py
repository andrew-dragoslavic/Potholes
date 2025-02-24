from ultralytics import YOLO
import cv2
from kaggle.api.kaggle_api_extended import KaggleApi
import kagglehub

api = KaggleApi()
api.authenticate()

# path = kagglehub.dataset_download("andrewmvd/pothole-detection")
# print(path)

model = YOLO("yolov8n.yaml")
model.train(data = "data.yaml", epochs = 50)

