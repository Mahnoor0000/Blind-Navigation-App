# Adds ~5,000 COCO images to merged_dataset_v3, using COCO's own (human) labels for the
# classes COCO has, and YOUR model for your classes COCO doesn't label (door, pole, stairs...).
#
# Why your model too: a COCO photo with a door in it has no door box, so YOLO would learn
# "door = background". Your model's confident boxes (conf >= OWN_CONF) fill that gap.
#
# Run from the FYP1 folder:
#   1. python scripts/add_coco.py      downloads labels + images into Dataset/coco_stage,
#                                      labels them, makes preview images. Nothing added yet.
#                                      Safe to stop and rerun: finished downloads are kept.
#   2. look at Dataset/coco_stage/preview (green = COCO label, orange = your model)
#   3. set APPLY = True and run again  -> copies into merged_dataset_v3
#
# COCO is CC BY 4.0. Cite: Lin et al., "Microsoft COCO: Common Objects in Context", ECCV 2014.

import os
import json
import random
import shutil
import hashlib
import zipfile
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import cv2
import yaml
from ultralytics import YOLO

# ---------------- settings ----------------
DST_DIR = "Dataset/merged_dataset_v3"
STAGE = "Dataset/coco_stage"
OWN_MODEL = "models/best.pt"   # your run4 model (29-class names are mapped to v3 below)
OWN_CONF = 0.5
PREFIX = "coco_"
APPLY = False
SEED = 0

# how many images to take per COCO class (an image can count for several), then fill up to N_TOTAL
PER_CLASS = {"cow": 600, "horse": 600, "sheep": 400, "stop sign": 700, "fire hydrant": 500,
             "dog": 500, "bench": 400, "traffic light": 500, "motorcycle": 500, "bicycle": 400,
             "bus": 300, "truck": 300, "suitcase": 200}
N_TOTAL = 5000
FILL_WITH = ["person", "car"]  # the rest: street photos with people / cars
SPLIT = {"train": 0.8, "valid": 0.1, "test": 0.1}
# ------------------------------------------

# COCO class name -> your v3 class name
COCO_TO_V3 = {
    "person": "person",
    "bicycle": "vehicle", "car": "vehicle", "motorcycle": "vehicle", "bus": "vehicle", "truck": "vehicle",
    "traffic light": "traffic_light", "fire hydrant": "fire_hydrant", "stop sign": "stop_sign",
    "parking meter": "pole", "bench": "bench", "dog": "dog",
    "cow": "animal", "horse": "animal", "sheep": "animal",
    "chair": "chair", "dining table": "table", "potted plant": "obstacle", "suitcase": "obstacle",
}
# your old (29-class) model names -> v3 names, ONLY for classes COCO does not label
OWN_FILL = {
    "pole": "pole", "tree": "tree", "stairs": "stairs", "door": "door", "shelf": "shelf", "bin": "bin",
    "fire_extinguisher": "fire_extinguisher", "obstacle": "obstacle", "branch": "obstacle",
    "pothole": "pothole", "crosswalk": "crosswalk", "traffic_cone": "traffic_cone", "elevator": "elevator",
}

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
os.makedirs(os.path.join(STAGE, "images"), exist_ok=True)
os.makedirs(os.path.join(STAGE, "labels"), exist_ok=True)

with open(os.path.join(DST_DIR, "data.yaml")) as f:
    v3_names = yaml.safe_load(f)["names"]
v3_id = {n: i for i, n in enumerate(v3_names)}

# 1. COCO labels (241 MB download, once)
ann_json = os.path.join(STAGE, "annotations", "instances_train2017.json")
if not os.path.exists(ann_json):
    zpath = os.path.join(STAGE, "annotations_trainval2017.zip")
    if not os.path.exists(zpath):
        print("downloading COCO labels (241 MB)...")
        urllib.request.urlretrieve(ANN_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extract("annotations/instances_train2017.json", STAGE)

print("reading COCO labels (needs ~2 GB RAM, takes a minute)...")
coco = json.load(open(ann_json))
cat_name = {c["id"]: c["name"] for c in coco["categories"]}
images = {im["id"]: im for im in coco["images"]}
anns = defaultdict(list)
for a in coco["annotations"]:
    anns[a["image_id"]].append(a)
del coco

# 2. choose images. Skip any image with a "crowd" box of a class we use
#    (a crowd box is one box over many people, which would leave them unlabelled).
usable = []
for iid, al in anns.items():
    if any(a["iscrowd"] and cat_name[a["category_id"]] in COCO_TO_V3 for a in al):
        continue
    usable.append(iid)
rng = random.Random(SEED)
rng.shuffle(usable)

chosen, per = [], Counter()
chosen_set = set()
for iid in usable:
    names = {cat_name[a["category_id"]] for a in anns[iid]}
    if any(n in PER_CLASS and per[n] < PER_CLASS[n] for n in names):
        chosen.append(iid)
        chosen_set.add(iid)
        for n in names:
            per[n] += 1
for iid in usable:
    if len(chosen) >= N_TOTAL:
        break
    if iid in chosen_set:
        continue
    names = {cat_name[a["category_id"]] for a in anns[iid]}
    if names & set(FILL_WITH):
        chosen.append(iid)
        chosen_set.add(iid)
chosen = chosen[:N_TOTAL]
print("images chosen:", len(chosen))
print("images per target class:", {k: per[k] for k in PER_CLASS})


# 3. download images (skips ones already downloaded)
def download(iid):
    im = images[iid]
    path = os.path.join(STAGE, "images", im["file_name"])
    if os.path.exists(path):
        return True
    try:
        urllib.request.urlretrieve(im["coco_url"], path + ".part")
        os.replace(path + ".part", path)
        return True
    except Exception:
        return False


print("downloading images...")
ok = 0
with ThreadPoolExecutor(16) as pool:
    for i, good in enumerate(pool.map(download, chosen), 1):
        ok += good
        if i % 500 == 0:
            print(f"  {i}/{len(chosen)}")
print("downloaded:", ok, "of", len(chosen))

# 4. labels: COCO boxes + your model's boxes for the classes COCO lacks
model = YOLO(OWN_MODEL)
own_map = {i: v3_id[OWN_FILL[n]] for i, n in model.names.items() if n in OWN_FILL}
box_count = Counter()
own_count = Counter()
labelled = []
for k, iid in enumerate(chosen, 1):
    im = images[iid]
    img_path = os.path.join(STAGE, "images", im["file_name"])
    if not os.path.exists(img_path):
        continue
    W, H = im["width"], im["height"]
    lines = []
    for a in anns[iid]:
        n = cat_name[a["category_id"]]
        if n not in COCO_TO_V3 or a["iscrowd"]:
            continue
        x, y, w, h = a["bbox"]
        if w < 4 or h < 4:
            continue
        c = v3_id[COCO_TO_V3[n]]
        lines.append(f"{c} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f} coco")
        box_count[COCO_TO_V3[n]] += 1
    r = model(img_path, conf=OWN_CONF, verbose=False)[0]
    for b in r.boxes:
        cls = int(b.cls)
        if cls in own_map:
            xc, yc, w, h = b.xywhn[0].tolist()
            lines.append(f"{own_map[cls]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} own")
            own_count[v3_names[own_map[cls]]] += 1
    with open(os.path.join(STAGE, "labels", os.path.splitext(im["file_name"])[0] + ".txt"), "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    labelled.append(im["file_name"])
    if k % 500 == 0:
        print(f"  labelled {k}/{len(chosen)}")

print("\nCOCO boxes added per class:", dict(box_count.most_common()))
print("your-model boxes added per class:", dict(own_count.most_common()))

# 5. preview: green = COCO label, orange = your model
prev_dir = os.path.join(STAGE, "preview")
os.makedirs(prev_dir, exist_ok=True)
for fn in random.Random(1).sample(labelled, min(30, len(labelled))):
    img = cv2.imread(os.path.join(STAGE, "images", fn))
    h, w = img.shape[:2]
    for line in open(os.path.join(STAGE, "labels", os.path.splitext(fn)[0] + ".txt")):
        c, xc, yc, bw, bh, src = line.split()
        xc, yc, bw, bh = float(xc) * w, float(yc) * h, float(bw) * w, float(bh) * h
        col = (0, 200, 0) if src == "coco" else (0, 140, 255)
        p1, p2 = (int(xc - bw / 2), int(yc - bh / 2)), (int(xc + bw / 2), int(yc + bh / 2))
        cv2.rectangle(img, p1, p2, col, 2)
        cv2.putText(img, v3_names[int(c)], (p1[0], max(12, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    cv2.imwrite(os.path.join(prev_dir, fn), img)
print("preview images in", prev_dir)


# 6. copy into merged_dataset_v3
def split_of(name):
    v = int(hashlib.md5(name.encode()).hexdigest(), 16) % 1000 / 1000
    acc = 0
    for s, frac in SPLIT.items():
        acc += frac
        if v < acc:
            return s
    return "train"


if APPLY:
    added, per_split = 0, Counter()
    for fn in labelled:
        s = split_of(fn)
        new = PREFIX + fn
        dst_img = os.path.join(DST_DIR, "images", s, new)
        if os.path.exists(dst_img):
            continue
        shutil.copy2(os.path.join(STAGE, "images", fn), dst_img)
        src_lbl = os.path.join(STAGE, "labels", os.path.splitext(fn)[0] + ".txt")
        with open(src_lbl) as f:
            lines = [" ".join(l.split()[:5]) for l in f if l.strip()]   # drop the coco/own tag
        with open(os.path.join(DST_DIR, "labels", s, os.path.splitext(new)[0] + ".txt"), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        added += 1
        per_split[s] += 1
    print(f"\nadded {added} images to {DST_DIR}:", dict(per_split))
    print("run count_classes.py (with v3) to see the new totals")
else:
    print("\nDRY RUN - nothing added to", DST_DIR, "- check the previews, then set APPLY = True")