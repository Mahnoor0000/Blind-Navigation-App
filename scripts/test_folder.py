# Runs the trained YOLO model on every image in test_images/
# Saves pictures with boxes drawn to test_results/ and a CSV you can fill in by hand.
#
# Run from the FYP1 folder:
#     python scripts/test_folder.py

import os
import csv
from collections import Counter
from ultralytics import YOLO

# ---- settings ----
model_path = "models/best.pt"     # your trained weights
image_dir = "test_images"         # folder with your test photos
out_dir = "test_results"          # where results go
conf = 0.25                       # minimum confidence to show a box
imgsz = 640                       # same size you trained with
# ------------------

image_types = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

if not os.path.exists(model_path):
    raise SystemExit("Model not found: " + model_path + "  (put best.pt in the models folder)")

images = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(image_types))
skipped = sorted(f for f in os.listdir(image_dir) if not f.lower().endswith(image_types))

if not images:
    raise SystemExit("No images found in " + image_dir)

os.makedirs(out_dir, exist_ok=True)
model = YOLO(model_path)
print("Model classes:", len(model.names))
print("Testing", len(images), "images\n")

total_counts = Counter()
rows = []

for name in images:
    path = os.path.join(image_dir, name)
    try:
        result = model.predict(path, conf=conf, imgsz=imgsz, verbose=False)[0]
    except Exception as e:
        print(name, "-> could not read:", e)
        continue

    # save the image with boxes drawn on it
    result.save(filename=os.path.join(out_dir, name.rsplit(".", 1)[0] + "_pred.jpg"))

    found = []
    for box in result.boxes:
        label = model.names[int(box.cls)]
        score = float(box.conf)
        found.append(label + " " + format(score, ".2f"))
        total_counts[label] += 1

    ms = result.speed["inference"]
    print(name)
    print("   ", ", ".join(found) if found else "nothing detected", "  (" + format(ms, ".0f") + " ms)")

    rows.append({
        "image": name,
        "detected": "; ".join(found) if found else "nothing",
        "num_boxes": len(found),
        "time_ms": round(ms),
        "missed (fill in)": "",
        "wrong label (fill in)": "",
        "false alarm (fill in)": "",
    })

# CSV for your manual check
csv_path = os.path.join(out_dir, "results.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print("\n---- summary ----")
print("images tested:", len(rows))
print("images with nothing detected:", sum(1 for r in rows if r["num_boxes"] == 0))
print("detections per class:")
for label, n in total_counts.most_common():
    print("   ", label.ljust(20), n)
if skipped:
    print("\nskipped (not an image type YOLO reads):", ", ".join(skipped))
print("\nPictures with boxes and results.csv saved in", out_dir)