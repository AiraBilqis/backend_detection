from ultralytics import YOLO

# Membuat model YOLOv8 dari awal atau menggunakan model pre-trained
model = YOLO('yolov8n.pt')  # Menggunakan model YOLOv8 pre-trained

# Melatih model menggunakan dataset yang sudah disiapkan
model.train(data='D:/Mobile IoT/detection_backend/datasets/dataset/data.yaml', epochs=50, imgsz=640)