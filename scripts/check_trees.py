import os
import random
from PIL import Image, ImageDraw

dataset_dir = "Dataset/merged_dataset_v2"
out_dir = "dataset_preview/tree_check"
tree_id = 4
how_many = 24

random.seed(1)
os.makedirs(out_dir, exist_ok=True)

areas = []
big = 0
small = 0
per_image_counts = []
examples = []

for split in ["train", "valid"]:
    lbl_dir = os.path.join(dataset_dir, "labels", split)
    img_dir = os.path.join(dataset_dir, "images", split)
    for fname in os.listdir(lbl_dir):
        if not fname.startswith("tree_"):
            continue
        boxes = []
        with open(os.path.join(lbl_dir, fname)) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 5 and int(parts[0]) == tree_id:
                    boxes.append([float(p) for p in parts[1:]])
        if len(boxes) == 0:
            continue
        per_image_counts.append(len(boxes))
        for b in boxes:
            area = b[2] * b[3]
            areas.append(area)
            if area > 0.5:
                big += 1
            if area < 0.01:
                small += 1
        stem = fname[:-4]
        for ext in [".jpg", ".jpeg", ".png"]:
            path = os.path.join(img_dir, stem + ext)
            if os.path.exists(path):
                examples.append((path, boxes))
                break

areas.sort()
print("tree boxes from the open images source:", len(areas))
print("images:", len(per_image_counts))
print("median box size (fraction of image):", round(areas[len(areas) // 2], 3))
print("boxes covering more than half the image:", big, "=", round(100 * big / len(areas)), "percent")
print("tiny boxes (under 1 percent of image):", small, "=", round(100 * small / len(areas)), "percent")
print("images with 5 or more tree boxes:", len([c for c in per_image_counts if c >= 5]))

# draw boxes on random images so you can look at them
random.shuffle(examples)
for i in range(min(how_many, len(examples))):
    path, boxes = examples[i]
    img = Image.open(path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    for b in boxes:
        x1 = (b[0] - b[2] / 2) * w
        y1 = (b[1] - b[3] / 2) * h
        x2 = (b[0] + b[2] / 2) * w
        y2 = (b[1] + b[3] / 2) * h
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=4)
    img.save(os.path.join(out_dir, str(i) + "_" + os.path.basename(path)))

print("saved", min(how_many, len(examples)), "images with boxes to", out_dir)