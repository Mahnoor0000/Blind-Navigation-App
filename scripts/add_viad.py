import os
import json
import random
from collections import Counter

import cv2
import numpy as np
import yaml

# ---------- settings ----------
viad_dir = r"D:\Downloads\archive\Senior-Design-VIAD-4"
out_dir = "Dataset/merged_dataset_v2"
prefix = "viad_"

# False = only count and make preview pictures, nothing is added to your dataset
# True  = really add the images and labels
apply_changes = True
preview_count = 24
preview_dir = "dataset_preview/viad_check"

train_part = 0.85
valid_part = 0.10
min_box_area = 0.0004      # ignore boxes smaller than this (fraction of the image)
noise_limit = 0.05         # a picture noisier than this gets cleaned

# new classes, added after the 20 you already have (the old ids do not change)
new_classes = [
    "pothole", "crosswalk", "traffic_cone", "elevator",
    "red_light", "green_light", "yellow_light", "bushes", "branch",
]

# name in this dataset -> name in your dataset. anything not listed is ignored
class_map = {
    "pothole": "pothole",
    "crosswalk": "crosswalk",
    "traffic_cone": "traffic_cone",
    "elevator": "elevator",
    "red_light": "red_light",
    "green_light": "green_light",
    "yellow_light": "yellow_light",
    "Bushes": "bushes",
    "branch": "branch",
    "tree": "tree",
    "door": "door",
    "stairs": "stairs",
    # cars, people and so on are kept so they do not turn into "empty background"
    "person": "person",
    "Person": "person",
    "car": "vehicle",
    "Bus": "vehicle",
    "Truck": "vehicle",
    "truck": "vehicle",
    "motorcycle": "vehicle",
    "train": "vehicle",
    "scooter": "cycle",
    "bicycle": "cycle",
    "bench": "bench",
    "fire_hydrant": "fire_hydrant",
    "stop_sign": "stop_sign",
    "chair": "chair",
    "traffic light": "traffic_light",
}

# images that contain these are not used at all
skip_images_with = ["gun", "rat"]

random.seed(42)

# ---------- step 1: the class list ----------
with open(os.path.join(out_dir, "data.yaml")) as f:
    info = yaml.safe_load(f)
names = list(info["names"])
for c in new_classes:
    if c not in names:
        names.append(c)
class_id = {}
for i in range(len(names)):
    class_id[names[i]] = i
for k in class_map:
    if class_map[k] not in class_id:
        print("class_map points to an unknown class:", class_map[k])
        raise SystemExit

# ---------- step 2: read the three coco files, group the copies of each photo ----------
copies = {}
for split in ["train", "valid", "test"]:
    json_path = os.path.join(viad_dir, split, "_annotations.coco.json")
    if not os.path.exists(json_path):
        print("missing:", json_path)
        continue
    with open(json_path) as f:
        data = json.load(f)

    cat_name = {}
    for c in data["categories"]:
        cat_name[c["id"]] = c["name"]

    boxes_by_image = {}
    for a in data["annotations"]:
        if a["image_id"] not in boxes_by_image:
            boxes_by_image[a["image_id"]] = []
        boxes_by_image[a["image_id"]].append((cat_name[a["category_id"]], a["bbox"]))

    for im in data["images"]:
        # roboflow adds ".rf.<hash>" to each copy of the same photo
        source = im["file_name"].split(".rf.")[0]
        path = os.path.join(viad_dir, split, im["file_name"])
        if source not in copies:
            copies[source] = []
        copies[source].append((path, im["width"], im["height"], boxes_by_image.get(im["id"], [])))

sources = sorted(copies.keys())
random.shuffle(sources)
print("unique photos:", len(sources), " image files:", sum(len(v) for v in copies.values()))


def noise_score(gray):
    # share of pixels that differ a lot from their neighbours (salt and pepper noise)
    smooth = cv2.medianBlur(gray, 3)
    return float(np.mean(cv2.absdiff(gray, smooth) > 60))


# ---------- step 3: pick one copy of every photo and convert the labels ----------
added = Counter()
class_boxes = Counter()
skipped_no_labels = 0
skipped_bad_class = 0
skipped_missing = 0
already_there = 0
cleaned = 0
previews = 0

if preview_count > 0:
    os.makedirs(preview_dir, exist_ok=True)

for n in range(len(sources)):
    source = sources[n]
    stem = prefix + source

    exists = False
    for s in ["train", "valid", "test"]:
        if os.path.exists(os.path.join(out_dir, "labels", s, stem + ".txt")):
            exists = True
    if exists:
        already_there += 1
        continue

    # the boxes are the same in every copy, so read them from the first one
    first_path, width, height, boxes = copies[source][0]
    lines = []
    bad = False
    for name, bbox in boxes:
        if name in skip_images_with:
            bad = True
        if name not in class_map:
            continue
        x, y, w, h = bbox
        x1 = max(0.0, x)
        y1 = max(0.0, y)
        x2 = min(float(width), x + w)
        y2 = min(float(height), y + h)
        if x2 <= x1 or y2 <= y1:
            continue
        if ((x2 - x1) * (y2 - y1)) / (width * height) < min_box_area:
            continue
        new_name = class_map[name]
        lines.append((class_id[new_name], (x1 + x2) / 2 / width, (y1 + y2) / 2 / height,
                      (x2 - x1) / width, (y2 - y1) / height))
    if bad:
        skipped_bad_class += 1
        continue
    if len(lines) == 0:
        skipped_no_labels += 1
        continue

    # use the least noisy copy
    best_path = None
    best_score = 1.0
    for path, w, h, b in copies[source]:
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        score = noise_score(gray)
        if score < best_score:
            best_score = score
            best_path = path
    if best_path is None:
        skipped_missing += 1
        continue

    image = cv2.imread(best_path)
    if best_score > noise_limit:
        image = cv2.medianBlur(image, 3)
        cleaned += 1

    position = n / len(sources)
    if position < train_part:
        split = "train"
    elif position < train_part + valid_part:
        split = "valid"
    else:
        split = "test"

    for line in lines:
        class_boxes[names[line[0]]] += 1
    added[split] += 1

    if previews < preview_count:
        big = cv2.resize(image, None, fx=2, fy=2)
        bh, bw = big.shape[:2]
        for line in lines:
            cx, cy, w, h = line[1] * bw, line[2] * bh, line[3] * bw, line[4] * bh
            p1 = (int(cx - w / 2), int(cy - h / 2))
            p2 = (int(cx + w / 2), int(cy + h / 2))
            cv2.rectangle(big, p1, p2, (0, 0, 255), 2)
            cv2.putText(big, names[line[0]], (p1[0], max(15, p1[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imwrite(os.path.join(preview_dir, str(previews) + "_" + stem + ".jpg"), big)
        previews += 1

    if apply_changes:
        cv2.imwrite(os.path.join(out_dir, "images", split, stem + ".jpg"), image,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        text = ""
        for line in lines:
            text = text + str(line[0]) + " " + " ".join(str(round(v, 6)) for v in line[1:]) + "\n"
        with open(os.path.join(out_dir, "labels", split, stem + ".txt"), "w") as f:
            f.write(text)

# ---------- step 4: write the new class list ----------
if apply_changes:
    info["names"] = names
    info["nc"] = len(names)
    with open(os.path.join(out_dir, "data.yaml"), "w") as f:
        yaml.safe_dump(info, f, sort_keys=False)

print()
print("images that would be added:", dict(added))
print("photos that needed noise cleaning:", cleaned)
print("skipped, no useful labels:", skipped_no_labels)
print("skipped, contains gun or rat:", skipped_bad_class)
print("skipped, image file missing:", skipped_missing)
print("skipped, already added earlier:", already_there)
print()
print("boxes per class from this dataset:")
for name in names:
    if class_boxes[name] > 0:
        print("   ", name.ljust(20), class_boxes[name])
print()
if apply_changes:
    print("CHANGES WRITTEN to", out_dir, "- data.yaml now has", len(names), "classes")
else:
    print("dry run only, nothing added. check", preview_dir, "then set apply_changes = True")