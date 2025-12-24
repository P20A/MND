from ultralytics import YOLO
import cv2
import time

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated = results[0].plot()

    cv2.imshow("YOLOv8 Test", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # time.sleep(2)

cap.release()
cv2.destroyAllWindows()
