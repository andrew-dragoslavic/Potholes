from ultralytics import YOLO
import cv2


# api = KaggleApi()
# api.authenticate()

# path = kagglehub.dataset_download("andrewmvd/pothole-detection")
# print(path)

model = YOLO("yolov8n.pt")
model.train(data = "data.yaml", epochs = 50)

