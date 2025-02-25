from ultralytics import YOLO
import cv2

cap = cv2.VideoCapture('test.mp4')
model = YOLO("runs/detect/train/weights/last.pt")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on the frame
    results = model.predict(source=frame, conf=0.4, verbose=False)
    
    # Extract bounding boxes and confidence scores
    boxes = results[0].boxes.xyxy.cpu().numpy()  # shape: [N, 4]
    scores = results[0].boxes.conf.cpu().numpy()   # shape: [N]

    # Draw bounding boxes and confidence scores on the frame
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Display the annotated frame
    cv2.imshow("YOLOv8 Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()