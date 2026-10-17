# Adds a single-class (or few-class) dataset, e.g. potholes, into merged_dataset_v2.
# Reads YOLO .txt labels AND Pascal VOC .xml labels.
#
# Problem it solves: a pothole dataset only labels potholes. People, cars, bikes etc. in
# those photos have no boxes, so YOLO would learn they are background. This script runs a
# COCO-pretrained YOLO over every new image and adds boxes for those classes too.
#
# Run from the FYP1 folder:
#   1. python scripts/add_dataset.py               (DRY RUN: counts + preview images only)
#   2. look at the images in Dataset/preview_added/
#   3. set APPLY = True and run again               (actually copies into the dataset)
#
# Safe to run twice: images that are already there are skipped.

import os
import random
import shutil
import hashlib
import xml.etree.ElementTree as ET
from collections import Counter

import cv2
import yaml
from ultralytics import YOLO

# ---------------- settings ----------------
# Set these for the dataset you are adding. Each setting appears ONCE - don't add copies.
SRC_DIR = "Dataset/stairs_autolabel"      # the dataset to add
DST_DIR = "Dataset/merged_dataset_v2"     # your merged dataset
PREFIX = "stairs3_"                       # added to file names so nothing clashes

# which source classes to keep, and which of YOUR class ids they become.
#   YOLO .txt labels -> key is the source class NUMBER, e.g. {0: 5}
#   VOC  .xml labels -> key is the source class NAME,   e.g. {"pothole": 20}
# Classes not listed are dropped (the image is still kept).
# Your ids: stairs = 5, pothole = 20
SRC_KEEP = {0: 5}                         # stairs

APPLY = True             # False = dry run, True = copy into the dataset
SKIP_UNLABELLED = True    # never add images that have no label file
AUTO_LABEL = True         # add person/vehicle/etc boxes with a COCO model
COCO_MODEL = "yolov8m.pt" # downloads automatically; use yolov8s.pt if too slow
COCO_CONF = 0.5           # only confident boxes, so few wrong labels get in
PREVIEW_COUNT = 25

# if the dataset has no train/valid/test folders, split it like this
SPLIT_IF_NONE = {"train": 0.8, "valid": 0.1, "test": 0.1}
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

# show the source class names (YOLO datasets) so SRC_KEEP can be checked
for root, _, files in os.walk(SRC_DIR):
    for fn in files:
        if fn in ("data.yaml", "dataset.yaml"):
            with open(os.path.join(root, fn)) as f:
                print("source classes (data.yaml):", yaml.safe_load(f).get("names"))
print("keeping:", {k: my_names[v] for k, v in SRC_KEEP.items()})
print()

# index every label file by name, wherever it is
txt_by_stem, xml_by_stem = {}, {}
for root, _, files in os.walk(SRC_DIR):
    for fn in files:
        stem, ext = os.path.splitext(fn)
        if ext.lower() == ".txt" and fn not in ("classes.txt", "README.txt", "done.txt"):
            txt_by_stem.setdefault(stem, []).append(os.path.join(root, fn))
        elif ext.lower() == ".xml":
            xml_by_stem.setdefault(stem, []).append(os.path.join(root, fn))


def split_from_path(path):
    parts = [p.lower() for p in path.replace("\\", "/").split("/")]
    for p in reversed(parts):
        if p in SPLIT_NAMES:
            return SPLIT_NAMES[p]
    return None


def split_by_hash(name):
    # same file always lands in the same split, so reruns are consistent
    v = int(hashlib.md5(name.encode()).hexdigest(), 16) % 1000 / 1000
    acc = 0
    for split, frac in SPLIT_IF_NONE.items():
        acc += frac
        if v < acc:
            return split
    return "train"


def pick_label(img_path, table):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    options = table.get(stem, [])
    if not options:
        return None
    if len(options) == 1:
        return options[0]
    # several with the same name: prefer the one next to the image (images -> labels)
    img_dir = os.path.dirname(img_path).replace("images", "labels")
    for o in options:
        if os.path.dirname(o) in (img_dir, os.path.dirname(img_path)):
            return o
    return options[0]


def read_txt(path, dropped):
    out = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            c = int(float(p[0]))
            if c not in SRC_KEEP:
                dropped[c] += 1
                continue
            v = [float(x) for x in p[1:]]
            if len(v) > 4:   # polygon -> box
                xs, ys = v[0::2], v[1::2]
                v = [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)]
            out.append((SRC_KEEP[c], *v[:4]))
    return out


def read_xml(path, img_path, dropped):
    root = ET.parse(path).getroot()
    size = root.find("size")
    W = H = 0
    if size is not None:
        W = int(float(size.findtext("width", "0")))
        H = int(float(size.findtext("height", "0")))
    if not W or not H:
        im = cv2.imread(img_path)
        if im is None:
            return []
        H, W = im.shape[:2]
    out = []
    for obj in root.iter("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name not in SRC_KEEP:
            dropped[name] += 1
            continue
        bb = obj.find("bndbox")
        x1, y1 = float(bb.findtext("xmin")), float(bb.findtext("ymin"))
        x2, y2 = float(bb.findtext("xmax")), float(bb.findtext("ymax"))
        x1, x2 = max(0, min(x1, x2)), min(W, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(H, max(y1, y2))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        out.append((SRC_KEEP[name], (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H))
    return out


# find all images
images = []
for root, _, files in os.walk(SRC_DIR):
    for fn in files:
        if fn.lower().endswith(IMG_TYPES):
            images.append(os.path.join(root, fn))
images.sort()

print("images found:", len(images))
print("label files found: .txt", sum(len(v) for v in txt_by_stem.values()),
      "| .xml", sum(len(v) for v in xml_by_stem.values()))

coco = YOLO(COCO_MODEL) if AUTO_LABEL else None

kept_boxes, auto_boxes, dropped_src = Counter(), Counter(), Counter()
split_count = Counter()
added = skipped_exist = skipped_unlabelled = no_kept_boxes = 0
preview_dir = "Dataset/preview_added"
os.makedirs(preview_dir, exist_ok=True)
preview_pick = set(random.Random(0).sample(range(len(images)), min(PREVIEW_COUNT, len(images))))

for i, img_path in enumerate(images):
    txt = pick_label(img_path, txt_by_stem)
    xml = pick_label(img_path, xml_by_stem)
    if txt is None and xml is None:
        if SKIP_UNLABELLED:
            skipped_unlabelled += 1
            continue

    new_name = PREFIX + os.path.basename(img_path)
    split = split_from_path(img_path) or split_by_hash(new_name)
    dst_img = os.path.join(DST_DIR, "images", split, new_name)
    dst_lbl = os.path.join(DST_DIR, "labels", split, os.path.splitext(new_name)[0] + ".txt")
    if APPLY and os.path.exists(dst_img):
        skipped_exist += 1
        continue

    boxes = []
    if txt:
        boxes = read_txt(txt, dropped_src)
    elif xml:
        boxes = read_xml(xml, img_path, dropped_src)
    if not boxes:
        no_kept_boxes += 1
    for b in boxes:
        kept_boxes[b[0]] += 1

    if coco is not None:
        r = coco.predict(img_path, conf=COCO_CONF, verbose=False)[0]
        for box in r.boxes:
            c = int(box.cls)
            if c in COCO_TO_MINE:
                x, y, w, h = box.xywhn[0].tolist()
                boxes.append((COCO_TO_MINE[c], x, y, w, h))
                auto_boxes[COCO_TO_MINE[c]] += 1

    lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in boxes]
    split_count[split] += 1

    if i in preview_pick:
        im = cv2.imread(img_path)
        if im is not None:
            H, W = im.shape[:2]
            src_ids = set(SRC_KEEP.values())
            for c, x, y, w, h in boxes:
                x, y, w, h = x * W, y * H, w * W, h * H
                color = (0, 200, 0) if c in src_ids else (0, 140, 255)
                cv2.rectangle(im, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), color, 2)
                cv2.putText(im, my_names[c], (int(x - w / 2), max(12, int(y - h / 2) - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.imwrite(os.path.join(preview_dir, os.path.splitext(new_name)[0] + ".jpg"), im)

    if APPLY:
        os.makedirs(os.path.dirname(dst_img), exist_ok=True)
        os.makedirs(os.path.dirname(dst_lbl), exist_ok=True)
        shutil.copy2(img_path, dst_img)
        with open(dst_lbl, "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        added += 1

    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{len(images)} done")

print()
print("images per split:", dict(split_count))
print("skipped, no label file:", skipped_unlabelled)
print("images with no kept box (kept as background):", no_kept_boxes)
print("boxes kept from the dataset:", {my_names[k]: v for k, v in kept_boxes.items()})
print("source labels dropped:", dict(dropped_src))
print("boxes added by the COCO model:", {my_names[k]: v for k, v in auto_boxes.most_common()})
print("preview images (green = dataset label, orange = auto label) in", preview_dir)
if APPLY:
    print("added", added, "images, skipped", skipped_exist, "already there")
else:
    print("DRY RUN - nothing copied. Check the numbers and previews, then set APPLY = True.")