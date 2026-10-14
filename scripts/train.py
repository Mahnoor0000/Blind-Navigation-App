from ultralytics import YOLO

data = "Dataset/merged_subset/data.yaml"
epochs = 15
imgsz = 416        # down from 640 — biggest single lever for CPU speed on 8GB RAM
batch = 8           # down from 16 — safer for 8GB RAM, avoid memory errors
device = "cpu"

model = YOLO("yolov8n.pt")

model.train(
    data=data,
    epochs=epochs,
    imgsz=imgsz,
    batch=batch,
    device=device
)