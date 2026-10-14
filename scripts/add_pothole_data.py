# Adds a single-class (or few-class) YOLO dataset, e.g. potholes, into merged_dataset_v2.
#
# Problem it solves: a pothole dataset only labels potholes. People, cars, bikes etc. in
# those photos have no boxes, so YOLO would learn they are background. This script runs a
# COCO-pretrained YOLO over every new image and adds boxes for those classes too.
#
# Run from the FYP1 folder:
#   1. python scripts/add_single_class_dataset.py          (DRY RUN: counts + preview images only)
#   2. look at the images in Dataset/preview_added/
#   3. set APPLY = True and run again                       (actually copies into the dataset)
#
# Safe to run twice: images that are already there are skipped.

import os
import random
import shutil
from collections import Counter

import cv2
import yaml
from ultralytics import YOLO

# ---------------- settings ----------------
SRC_DIR = r"D:\Downloads\pothole data"      # the downloaded dataset folder
DST_DIR = "Dataset/merged_dataset_v2"        # your merged dataset
PREFIX = "pot_"                              # added to file names so nothing clashes

# source class id -> your class id. Classes not listed here are dropped.
# Check the printed source class names first, then fill this in.
# Your ids: pothole = 20
SRC_KEEP = {0: 20}

APPLY = True             # False = dry run, True = copy into the dataset
AUTO_LABEL = True         # add person/vehicle/etc boxes with a COCO model
COCO_MODEL = "yolov8m.pt" # downloads automatically; use yolov8s.pt if too slow
COCO_CONF = 0.5           # only confident boxes, so few wrong labels get in
PREVIEW_COUNT = 25
# ------------------------------------------

# COCO class id -> your class id (only classes that matter for walking)
COCO_TO_MINE = {
    0: 0,    # person -> person
    1: 1,    # bicycle -> vehicle (cycle was merged into vehicle)
    2: 1,    # car -> vehicle
    3: 1,    # motorcycle -> vehicle
    5: 1,    # bus -> vehicle
    7: 1,    # truck -> vehicle
    9: 13,   # traffic light
    10: 15,  # fire hydrant
    11: 14,  # stop sign
    12: 17,  # parking meter
    13: 16,  # bench
    16: 19,  # dog
}

IMG_TYPES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SPLIT_NAMES = {"train": "train", "val": "valid", "valid": "valid", "test": "test"}

with open(os.path.join(DST_DIR, "data.yaml")) as f:
    my_names = yaml.safe_load(f)["names"]
if isinstance(my_names, dict):
    my_names = [my_names[k] for k in sorted(my_names)]

# show the source class names so SRC_KEEP can be checked
for root, _, files in os.walk(SRC_DIR):
    for fn in files:
        if fn in ("data.yaml", "dataset.yaml"):
            with open(os.path.join(root, fn)) as f:
                print("source classes:", yaml.safe_load(f).get("names"))
print("keeping:", {k: my_names[v] for k, v in SRC_KEEP.items()})
print()


def split_of(path):
    parts = [p.lower() for p in path.replace("\\", "/").split("/")]
    for p in reversed(parts):
        if p in SPLIT_NAMES:
            return SPLIT_NAMES[p]
    return "train"


# find every (image, label file) pair, for both layouts:
#   train/images + train/labels   and   images/train + labels/train
pairs = []
for root, _, files in os.walk(SRC_DIR):
    for fn in files:
        if not fn.lower().endswith(IMG_TYPES):
            continue
        img = os.path.join(root, fn)
        stem = os.path.splitext(fn)[0]
        label_dir = root.replace("images", "labels")
        lbl = os.path.join(label_dir, stem + ".txt")
        pairs.append((img, lbl if os.path.exists(lbl) else None, split_of(root)))

print("images found:", len(pairs))
print("without a label file:", sum(1 for _, l, _ in pairs if l is None), "(kept as background)")
print("per split:", dict(Counter(s for _, _, s in pairs)))
print()

coco = YOLO(COCO_MODEL) if AUTO_LABEL else None

kept_boxes = Counter()
auto_boxes = Counter()
dropped_src = Counter()
added = skipped = 0
preview_dir = "Dataset/preview_added"
os.makedirs(preview_dir, exist_ok=True)
preview_pick = set(random.Random(0).sample(range(len(pairs)), min(PREVIEW_COUNT, len(pairs))))

for i, (img_path, lbl_path, split) in enumerate(pairs):
    new_name = PREFIX + os.path.basename(img_path)
    dst_img = os.path.join(DST_DIR, "images", split, new_name)
    dst_lbl = os.path.join(DST_DIR, "labels", split, os.path.splitext(new_name)[0] + ".txt")
    if APPLY and os.path.exists(dst_img):
        skipped += 1
        continue

    lines = []
    if lbl_path:
        with open(lbl_path) as f:
            for line in f:
                p = line.split()
                if len(p) < 5:
                    continue
                c = int(p[0])
                if c in SRC_KEEP:
                    if len(p) > 5:   # polygon -> box
                        xs = [float(v) for v in p[1::2]]
                        ys = [float(v) for v in p[2::2]]
                        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
                        p = [p[0], (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
                    lines.append(f"{SRC_KEEP[c]} {float(p[1]):.6f} {float(p[2]):.6f} {float(p[3]):.6f} {float(p[4]):.6f}")
                    kept_boxes[SRC_KEEP[c]] += 1
                else:
                    dropped_src[c] += 1

    if coco is not None:
        r = coco.predict(img_path, conf=COCO_CONF, verbose=False)[0]
        for box in r.boxes:
            c = int(box.cls)
            if c in COCO_TO_MINE:
                x, y, w, h = box.xywhn[0].tolist()
                lines.append(f"{COCO_TO_MINE[c]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                auto_boxes[COCO_TO_MINE[c]] += 1

    if i in preview_pick:
        im = cv2.imread(img_path)
        if im is not None:
            H, W = im.shape[:2]
            for line in lines:
                c, x, y, w, h = line.split()
                c = int(c)
                x, y, w, h = float(x) * W, float(y) * H, float(w) * W, float(h) * H
                color = (0, 200, 0) if c in SRC_KEEP.values() else (0, 140, 255)
                cv2.rectangle(im, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), color, 2)
                cv2.putText(im, my_names[c], (int(x - w / 2), int(y - h / 2) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.imwrite(os.path.join(preview_dir, new_name.rsplit(".", 1)[0] + ".jpg"), im)

    if APPLY:
        os.makedirs(os.path.dirname(dst_img), exist_ok=True)
        os.makedirs(os.path.dirname(dst_lbl), exist_ok=True)
        shutil.copy2(img_path, dst_img)
        with open(dst_lbl, "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        added += 1

    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{len(pairs)} done")

print()
print("boxes kept from the dataset:", {my_names[k]: v for k, v in kept_boxes.items()})
print("boxes dropped (source ids not in SRC_KEEP):", dict(dropped_src))
print("boxes added by the COCO model:", {my_names[k]: v for k, v in auto_boxes.most_common()})
print("preview images (green = dataset label, orange = auto label) in", preview_dir)
if APPLY:
    print("added", added, "images, skipped", skipped, "already there")
else:
    print("DRY RUN - nothing copied. Check the previews, then set APPLY = True.")