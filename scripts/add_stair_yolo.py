import os
import random
import shutil
import yaml
from PIL import Image, ImageOps

# folder where you extracted the second stairs dataset (images + txt labels)
stairs_dir = r"D:\Downloads\stairs"

# the merged 20 class dataset
out_dir = "Dataset/merged_dataset_v2"

stairs_id = 5          # id of "stairs" in the 20 class list
prefix = "stairs2_"    # so the file names never clash with the first stairs dataset
max_side = 960
train_part = 0.85
valid_part = 0.10

random.seed(42)

# step 1: find images, label files and the class names (if the dataset has them)
images = {}
labels = {}
names = None
for root, folders, files in os.walk(stairs_dir):
    for fname in files:
        low = fname.lower()
        full = os.path.join(root, fname)
        stem = os.path.splitext(fname)[0]
        if low.endswith((".jpg", ".jpeg", ".png")):
            images[stem] = full
        elif low.endswith(".txt") and low != "classes.txt":
            labels[stem] = full
        elif low in ("data.yaml", "dataset.yaml"):
            with open(full) as f:
                info = yaml.safe_load(f)
            names = info["names"]
            if isinstance(names, dict):
                names = [names[k] for k in sorted(names)]
        elif low == "classes.txt" and names is None:
            with open(full) as f:
                names = [line.strip() for line in f if line.strip()]

print("images found:", len(images), " label files found:", len(labels))
print("class names in dataset:", names)

# step 2: work out which old ids mean stairs
id_map = {}
if names is not None:
    for i in range(len(names)):
        if "stair" in names[i].lower():
            id_map[i] = stairs_id
else:
    # no names file: only safe if every label uses class 0
    all_ids = set()
    for stem in labels:
        with open(labels[stem]) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 5:
                    all_ids.add(int(parts[0]))
    print("class ids used in labels:", all_ids)
    if all_ids != {0}:
        print("more than one class id and no names file, stopping. send me the ids")
        raise SystemExit
    id_map[0] = stairs_id

print("id map (old id -> new id):", id_map)
if len(id_map) == 0:
    print("no class with 'stair' in its name, stopping")
    raise SystemExit

# step 3: convert and add
stems = sorted(labels.keys())
random.shuffle(stems)

added = {"train": 0, "valid": 0, "test": 0}
no_image = 0
no_stairs = 0
already_there = 0

for n in range(len(stems)):
    stem = stems[n]
    if stem not in images:
        no_image += 1
        continue

    new_stem = prefix + stem
    exists = False
    for s in ["train", "valid", "test"]:
        if os.path.exists(os.path.join(out_dir, "labels", s, new_stem + ".txt")):
            exists = True
    if exists:
        already_there += 1
        continue

    new_lines = []
    with open(labels[stem]) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            old_id = int(parts[0])
            if old_id in id_map:
                new_lines.append(str(id_map[old_id]) + " " + " ".join(parts[1:]))

    if len(new_lines) == 0:
        no_stairs += 1
        continue

    position = n / len(stems)
    if position < train_part:
        split = "train"
    elif position < train_part + valid_part:
        split = "valid"
    else:
        split = "test"

    img = Image.open(images[stem])
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))
    img.save(os.path.join(out_dir, "images", split, new_stem + ".jpg"), quality=92)

    with open(os.path.join(out_dir, "labels", split, new_stem + ".txt"), "w") as f:
        f.write("\n".join(new_lines) + "\n")
    added[split] += 1

print("added:", added)
print("skipped, no matching image:", no_image)
print("skipped, no stairs box:", no_stairs)
print("skipped, already added earlier:", already_there)