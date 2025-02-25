from ultralytics import YOLO

model = YOLO("runs/detect/train7/weights/last.pt")

results = model.track(source="test.mp4", conf = 0.6, show = True)