import os
import random
import shutil
import yaml

# existing merged dataset (13 classes) and where the new 20 class version goes
base_dir = "Dataset/merged_dataset"
out_dir = "Dataset/merged_dataset_v2"

# first 13 keep the same ids as before, the 7 new ones are added at the end
unified_classes = [
    "person", "vehicle", "cycle", "pole", "tree", "stairs",
    "door", "chair", "table", "shelf", "bin", "fire_extinguisher", "obstacle",
    "traffic_light", "stop_sign", "fire_hydrant", "bench",
    "parking_meter", "potted_plant", "dog",
]
unified_id = {}
for i in range(len(unified_classes)):
    unified_id[unified_classes[i]] = i

coco_map = {
    "traffic light": "traffic_light",
    "stop sign": "stop_sign",
    "fire hydrant": "fire_hydrant",
    "bench": "bench",
    "parking meter": "parking_meter",
    "potted plant": "potted_plant",
    "dog": "dog",
    "bicycle": "cycle",
    "person": "person",
    "car": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
    "motorcycle": "vehicle",
    "chair": "chair",
    "dining table": "table",
}

tree_map = {
    "Tree": "tree",
    "Person": "person",
    "Car": "vehicle",
    "Bus": "vehicle",
    "Truck": "vehicle",
    "Motorcycle": "vehicle",
    "Bicycle": "cycle",
    "Chair": "chair",
    "Table": "table",
}

# (folder, name map, prefix for file names)
sources = [
    ("Dataset/coco_subset", coco_map, "coco"),
    ("Dataset/trees_subset", tree_map, "tree"),
]

# how the new images are split between train / valid / test
train_part = 0.85
valid_part = 0.10

random.seed(42)


def read_names(folder):
    yaml_path = os.path.join(folder, "dataset.yaml")
    if not os.path.exists(yaml_path):
        yaml_path = os.path.join(folder, "data.yaml")
    with open(yaml_path) as f:
        info = yaml.safe_load(f)
    names = info["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return names


def find_pairs(folder):
    # collect every (image, label) pair from all split folders
    pairs = []
    images_root = os.path.join(folder, "images")
    labels_root = os.path.join(folder, "labels")
    for split in os.listdir(images_root):
        img_dir = os.path.join(images_root, split)
        lbl_dir = os.path.join(labels_root, split)
        if not os.path.isdir(img_dir):
            continue
        for fname in os.listdir(img_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            stem = os.path.splitext(fname)[0]
            lbl_path = os.path.join(lbl_dir, stem + ".txt")
            if os.path.exists(lbl_path):
                pairs.append((os.path.join(img_dir, fname), lbl_path))
    return pairs


# step 1: copy the existing merged dataset so the original stays untouched
if os.path.exists(out_dir):
    print("output folder already exists, delete it first:", out_dir)
    raise SystemExit
shutil.copytree(base_dir, out_dir)
print("copied", base_dir, "to", out_dir)

# step 2: add each new source
for folder, name_map, prefix in sources:
    names = read_names(folder)

    # old id in this source -> new unified id (None if we do not use it)
    id_map = {}
    for i in range(len(names)):
        if names[i] in name_map:
            id_map[i] = unified_id[name_map[names[i]]]

    pairs = find_pairs(folder)
    random.shuffle(pairs)

    added = {"train": 0, "valid": 0, "test": 0}
    skipped = 0

    for n in range(len(pairs)):
        img_path, lbl_path = pairs[n]

        # rewrite the label lines with the new ids
        new_lines = []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                old_id = int(parts[0])
                if old_id in id_map:
                    new_lines.append(str(id_map[old_id]) + " " + " ".join(parts[1:]))

        # image has none of the classes we use, skip it
        if len(new_lines) == 0:
            skipped += 1
            continue

        position = n / len(pairs)
        if position < train_part:
            split = "train"
        elif position < train_part + valid_part:
            split = "valid"
        else:
            split = "test"

        img_name = prefix + "_" + os.path.basename(img_path)
        stem = os.path.splitext(img_name)[0]

        shutil.copy2(img_path, os.path.join(out_dir, "images", split, img_name))
        with open(os.path.join(out_dir, "labels", split, stem + ".txt"), "w") as f:
            f.write("\n".join(new_lines) + "\n")
        added[split] += 1

    print(prefix, "added:", added, "skipped (no useful labels):", skipped)

# step 3: write the new data.yaml with all 20 classes
data = {
    "path": os.path.abspath(out_dir),
    "train": "images/train",
    "val": "images/valid",
    "test": "images/test",
    "nc": len(unified_classes),
    "names": unified_classes,
}
with open(os.path.join(out_dir, "data.yaml"), "w") as f:
    yaml.safe_dump(data, f, sort_keys=False)

print("done. new dataset:", out_dir)