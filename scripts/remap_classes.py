# Builds merged_dataset_v3 from merged_dataset_v2 with fewer, cleaner classes:
#   red / green / yellow_light -> traffic_light   (colours were confused; app says "traffic light" anyway)
#   parking_meter              -> pole
#   cycle (0), bushes, tree, branch, potted_plant -> removed
#     (v3 first merged branch + potted_plant into obstacle: obstacle fell from 0.86 to 0.27 mAP50,
#      so obstacle is kept clean now; tree was 0.31 mAP50 and the app ignores it)
#   animal                     -> new, empty for now (COCO cows / horses / sheep get added later)
#
# merged_dataset_v2 is NOT changed. Images are hard-linked (no extra disk space) or copied.
#
# Run from the FYP1 folder:
#   python scripts/remap_classes.py        (DRY RUN: prints old -> new class counts)
#   set APPLY = True and run again          (writes Dataset/merged_dataset_v3)

import os
import shutil
from collections import Counter

import yaml

# ---------------- settings ----------------
SRC_DIR = "Dataset/merged_dataset_v2"
DST_DIR = "Dataset/merged_dataset_v3"
APPLY = True

# new class list, in id order (0, 1, 2, ...)
NEW_NAMES = [
    "person", "vehicle", "pole", "stairs", "door", "chair", "table", "shelf", "bin",
    "fire_extinguisher", "obstacle", "traffic_light", "stop_sign", "fire_hydrant", "bench",
    "dog", "pothole", "crosswalk", "traffic_cone", "elevator", "animal",
]

# old class name -> new class name (None = delete those boxes). Classes not listed keep their name.
CHANGE = {
    "cycle": None,
    "bushes": None,
    "tree": None,          # to keep trees: delete this line and put "tree" back in NEW_NAMES
    "red_light": "traffic_light",
    "green_light": "traffic_light",
    "yellow_light": "traffic_light",
    "parking_meter": "pole",
    "branch": None,
    "potted_plant": None,
}
# ------------------------------------------

SPLITS = ["train", "valid", "test"]
IMG_TYPES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

with open(os.path.join(SRC_DIR, "data.yaml")) as f:
    old_names = yaml.safe_load(f)["names"]
if isinstance(old_names, dict):
    old_names = [old_names[k] for k in sorted(old_names)]

# old id -> new id (or None)
id_map = {}
for i, name in enumerate(old_names):
    new = CHANGE.get(name, name)
    if new is None:
        id_map[i] = None
    elif new in NEW_NAMES:
        id_map[i] = NEW_NAMES.index(new)
    else:
        raise SystemExit(f"class '{name}' -> '{new}' is not in NEW_NAMES; fix CHANGE or NEW_NAMES")

print("old -> new:")
for i, name in enumerate(old_names):
    print(f"  {i:2d} {name:18s} -> {NEW_NAMES[id_map[i]] if id_map[i] is not None else 'DELETED'}")

old_count = {s: Counter() for s in SPLITS}
new_count = {s: Counter() for s in SPLITS}
emptied = 0
n_files = 0

for split in SPLITS:
    lbl_dir = os.path.join(SRC_DIR, "labels", split)
    img_dir = os.path.join(SRC_DIR, "images", split)
    if not os.path.isdir(lbl_dir):
        continue
    out_lbl = os.path.join(DST_DIR, "labels", split)
    out_img = os.path.join(DST_DIR, "images", split)
    if APPLY:
        os.makedirs(out_lbl, exist_ok=True)
        os.makedirs(out_img, exist_ok=True)

    # labels
    for fname in os.listdir(lbl_dir):
        if not fname.endswith(".txt"):
            continue
        n_files += 1
        with open(os.path.join(lbl_dir, fname)) as f:
            lines = [l.split() for l in f if l.strip()]
        out = []
        for p in lines:
            old = int(float(p[0]))
            old_count[split][old] += 1
            new = id_map.get(old)
            if new is None:
                continue
            new_count[split][new] += 1
            out.append(" ".join([str(new)] + p[1:]))
        if lines and not out:
            emptied += 1                 # only had deleted classes -> now a background image
        if APPLY:
            with open(os.path.join(out_lbl, fname), "w") as f:
                f.write("\n".join(out) + ("\n" if out else ""))

    # images (hard link = no extra space; falls back to copy)
    if APPLY:
        for fname in os.listdir(img_dir):
            if not fname.lower().endswith(IMG_TYPES):
                continue
            dst = os.path.join(out_img, fname)
            if os.path.exists(dst):
                continue
            try:
                os.link(os.path.join(img_dir, fname), dst)
            except OSError:
                shutil.copy2(os.path.join(img_dir, fname), dst)
    print(f"{split}: done")

print(f"\nlabel files: {n_files} | images left with no boxes (now background): {emptied}")
print(f"\n{'id':>3} {'class':18s} {'train':>7} {'valid':>7} {'test':>7} {'total':>7}")
for i, name in enumerate(NEW_NAMES):
    t, v, s = (new_count[sp][i] for sp in SPLITS)
    print(f"{i:>3} {name:18s} {t:>7} {v:>7} {s:>7} {t + v + s:>7}")
print("total boxes before:", sum(sum(c.values()) for c in old_count.values()),
      "| after:", sum(sum(c.values()) for c in new_count.values()))

if APPLY:
    with open(os.path.join(DST_DIR, "data.yaml"), "w") as f:
        yaml.safe_dump({"path": os.path.abspath(DST_DIR).replace("\\", "/"),
                        "train": "images/train", "val": "images/valid", "test": "images/test",
                        "nc": len(NEW_NAMES), "names": NEW_NAMES}, f, sort_keys=False)
    print("\nwritten:", DST_DIR, "(merged_dataset_v2 unchanged)")
else:
    print("\nDRY RUN - nothing written. Set APPLY = True to create", DST_DIR)