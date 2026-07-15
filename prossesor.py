from ultralytics import YOLO
import cv2
import time

CONF_THRESHOLD = 0.65
MIN_DIGITS = 5
RETRY_DELAY = 0.5     

model = YOLO(r"D:\image procesing\program\best.pt")
cap = cv2.VideoCapture(0)

def capture_meter_number():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed, retrying...")
            time.sleep(RETRY_DELAY)
            continue

        results = model(frame)
        boxes = results[0].boxes

        digits = []
        valid = True

        if boxes is None or len(boxes) < MIN_DIGITS:
            valid = False

        else:
            for box in boxes:
                conf = float(box.conf[0])
                if conf < CONF_THRESHOLD:
                    valid = False
                    break

                x1 = box.xyxy[0][0].item()
                digit = int(box.cls[0].item())
                digits.append((x1, digit))

        annotated = results[0].plot()
        cv2.imshow("Meter Capture", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            return None

        if not valid:
            print("Retrying... (low confidence or missing digits)")
            time.sleep(RETRY_DELAY)
            continue

        digits.sort(key=lambda x: x[0])
        meter_number = ''.join(str(d[1]) for d in digits)

        print("✅ Final meter reading:", meter_number)
        return meter_number


try:
    while True:
        meter_value = capture_meter_number()
finally:
    cap.release()
    cv2.destroyAllWindows()
