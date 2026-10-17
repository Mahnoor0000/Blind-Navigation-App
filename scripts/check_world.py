import os
from ultralytics import YOLO

SRC_DIR = r"D:\Downloads\new stair data\stairs_normal_dataset\stairs"

model = YOLO("yolov8s-worldv2.pt")
model.set_classes(["staircase", "stairs", "steps"])
print("classes:", model.names)

files = [f for f in os.listdir(SRC_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))][:5]
for f in files:
    r = model.predict(os.path.join(SRC_DIR, f), conf=0.01, verbose=False)[0]
    scores = sorted((float(c) for c in r.boxes.conf), reverse=True)[:3]
    print(f, "->", len(r.boxes), "boxes, top scores:", [round(s, 2) for s in scores])