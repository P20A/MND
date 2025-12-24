from ultralytics import YOLO
model = YOLO("yolov8n.pt")

data_path = 'D:\image procesing\program\digit-meter-recognition.v5i.yolov8\data.yaml'
model.train(data=data_path,epochs=50,imgsz=640)