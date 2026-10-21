# Makes a training subset of merged_dataset_v3 for Colab, WITHOUT capping big classes.
#
# Old make_subset used a per-class quota (max 800), which threw away person/vehicle images
# and dropped their recall. Here:
#   1. every class first gets at least FLOOR boxes (rare classes are picked first)
#   2. the rest of the subset is filled with random images, so person / vehicle keep
#      their natural share
#
# Run from the FYP1 folder:
#   python scripts/make_subset_v3.py
# Then upload Dataset/merged_subset_v3.zip to Google Drive, folder FYP/.

import os
import random
import shutil
from collections import Counter

import yaml

# ---------------- settings ----------------
SRC_DIR = "Dataset/merged_dataset_v3"
DST_DIR = "Dataset/merged_subset_v3"
N = {"train": 12000, "valid": 1000, "test": 700}   # images per split
FLOOR = {"train": 1000, "valid": 60, "test": 40}  # min boxes per class (or all it has)
SEED = 0
# ------------------------------------------

rng = random.Random(SEED)
with open(os.path.join(SRC_DIR, "data.yaml")) as f:
    names = yaml.safe_load(f)["names"]

if os.path.exists(DST_DIR):
    shutil.rmtree(DST_DIR)

print(f"{'class':18s} " + " ".join(f"{s:>7}" for s in N))
totals = {}
for split, n_target in N.items():
    lbl_dir = os.path.join(SRC_DIR, "labels", split)
    img_dir = os.path.join(SRC_DIR, "images", split)
    imgs = {os.path.splitext(f)[0]: f for f in os.listdir(img_dir)}
    boxes = {}
    for fn in os.listdir(lbl_dir):
        stem = os.path.splitext(fn)[0]
        if stem not in imgs:
            continue
        with open(os.path.join(lbl_dir, fn)) as f:
            boxes[stem] = Counter(int(l.split()[0]) for l in f if l.strip())
    stems = list(boxes)
    rng.shuffle(stems)

    chosen, got = set(), Counter()
    # 1. floor for each class, rarest class first
    total_per_class = Counter()
    for c in boxes.values():
        total_per_class.update(c)
    for cls in sorted(range(len(names)), key=lambda c: total_per_class[c]):
        need = min(FLOOR[split], total_per_class[cls])
        for s in stems:
            if got[cls] >= need or len(chosen) >= n_target:
                break
            if s not in chosen and boxes[s][cls]:
                chosen.add(s)
                got.update(boxes[s])
    # 2. fill randomly
    for s in stems:
        if len(chosen) >= n_target:
            break
        if s not in chosen:
            chosen.add(s)
            got.update(boxes[s])

    os.makedirs(os.path.join(DST_DIR, "images", split))
    os.makedirs(os.path.join(DST_DIR, "labels", split))
    for s in chosen:
        shutil.copy2(os.path.join(img_dir, imgs[s]), os.path.join(DST_DIR, "images", split, imgs[s]))
        shutil.copy2(os.path.join(lbl_dir, s + ".txt"), os.path.join(DST_DIR, "labels", split, s + ".txt"))
    totals[split] = (len(chosen), got)

for i, name in enumerate(names):
    print(f"{name:18s} " + " ".join(f"{totals[s][1][i]:>7}" for s in N))
print("images:", {s: totals[s][0] for s in N})

with open(os.path.join(DST_DIR, "data.yaml"), "w") as f:
    yaml.safe_dump({"path": ".", "train": "images/train", "val": "images/valid", "test": "images/test",
                    "nc": len(names), "names": names}, f, sort_keys=False)

print("zipping...")
z = shutil.make_archive(DST_DIR, "zip", DST_DIR)
print("upload this to Google Drive, folder FYP/:", z)