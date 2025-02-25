from ultralytics import YOLO
import numpy as np
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

cap = cv2.VideoCapture('test.mp4')
model = YOLO("runs/detect/train/weights/last.pt")

tracker = DeepSort(max_age=30, n_init=3, embedder="mobilenet")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 4. Run YOLOv8 inference on the frame
    results = model.predict(source=frame, conf=0.3, verbose=False)
    
    # 5. Extract bounding boxes and confidence scores
    # Assumption: results[0].boxes.xyxy is a tensor of shape [N,4] (x1, y1, x2, y2)
    # and results[0].boxes.conf is a tensor of shape [N]
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Coordinates in float
    scores = results[0].boxes.conf.cpu().numpy()   # Confidence scores

    # 6. Prepare detections for DeepSORT:
    # DeepSORT expects a list of detections in the format: [x1, y1, x2, y2, score, class_id]
    # If you have one class (pothole), you can set class_id to 0 for all.
    detections = []
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        detections.append([[x1, y1, x2, y2], score, 0])
    
    # 7. Update the tracker with the current frame detections
    tracks = tracker.update_tracks(detections, frame=frame)
    
    # 8. Draw bounding boxes and track IDs on the frame
    for track in tracks:
        # Skip unconfirmed tracks
        if not track.is_confirmed():
            continue
        
        track_id = track.track_id
        # DeepSORT returns the bounding box in left, top, right, bottom format
        ltrb = track.to_ltrb()  
        x1, y1, x2, y2 = map(int, ltrb)
        
        # Draw the bounding box and unique ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    
    # 9. Display the frame
    cv2.imshow("YOLOv8 + DeepSORT Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()