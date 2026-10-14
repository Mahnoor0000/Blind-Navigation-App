import os
import json
import yaml
from collections import Counter

# change this to the folder you want to check
dataset_dir = "Dataset/merged_dataset_v2"

print("top level of", dataset_dir)
for name in os.listdir(dataset_dir):
    print("   ", name)


def split_of(path):
    # which split (train / valid / test) a folder belongs to
    parts = path.replace("\\", "/").lower().split("/")
    for p in parts:
        if p in ("train", "valid", "val", "test"):
            if p == "val":
                return "valid"
            return p
    return "all"


# class names from the dataset's own yaml file, if it has one
names = None
for fname in ("data.yaml", "dataset.yaml"):
    yaml_path = os.path.join(dataset_dir, fname)
    if os.path.exists(yaml_path):
        with open(yaml_path) as f:
            info = yaml.safe_load(f)
        names = info["names"]
        # names can be a list or a dict like {0: "person", 1: "car"}
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]

counts = {}          # class -> Counter of boxes per split
images = Counter()   # split -> number of image files
sources = {}         # split -> set of original photo names
label_files = Counter()
long_lines = 0
coco_files = []

for root, folders, files in os.walk(dataset_dir):
    split = split_of(os.path.relpath(root, dataset_dir))
    # works for both layouts: labels/train/x.txt and train/labels/x.txt
    in_labels = "labels" in os.path.relpath(root, dataset_dir).replace("\\", "/").lower().split("/")
    for fname in files:
        low = fname.lower()
        path = os.path.join(root, fname)

        if low.endswith((".jpg", ".jpeg", ".png")):
            images[split] += 1
            # roboflow adds ".rf.<hash>" to every copy of the same photo
            source = fname
            if ".rf." in fname:
                source = fname.split(".rf.")[0]
            if split not in sources:
                sources[split] = set()
            sources[split].add(source)

        elif low.endswith(".txt") and in_labels:
            label_files[split] += 1
            with open(path) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    if len(parts) > 5:
                        long_lines += 1
                    class_id = int(parts[0])
                    if class_id not in counts:
                        counts[class_id] = Counter()
                    counts[class_id][split] += 1

        elif low.endswith(".json") and ("annotation" in low or "coco" in low):
            coco_files.append((path, split))

print()
print("split    images   unique photos   label files")
for split in sorted(images):
    print(split.ljust(8), str(images[split]).ljust(8), str(len(sources[split])).ljust(15), label_files[split])

# COCO json export
for path, split in coco_files:
    with open(path) as f:
        data = json.load(f)
    cat_names = {}
    for c in data["categories"]:
        cat_names[c["id"]] = c["name"]
    per_class = Counter()
    for a in data["annotations"]:
        per_class[a["category_id"]] += 1
    print()
    print("COCO file:", path, "(", len(data["images"]), "images )")
    for cid in sorted(cat_names):
        print("   ", cat_names[cid].ljust(20), per_class[cid])

# YOLO txt export
if len(counts) > 0:
    print()
    print("YOLO boxes per class")
    print("class".ljust(24), "train".ljust(8), "valid".ljust(8), "test".ljust(8), "total")
    for class_id in sorted(counts):
        if names is not None and class_id < len(names):
            label = names[class_id]
        else:
            label = "id " + str(class_id)
        c = counts[class_id]
        total = sum(c.values())
        print(label.ljust(24), str(c["train"]).ljust(8), str(c["valid"]).ljust(8), str(c["test"]).ljust(8), total)
    if long_lines > 0:
        print("note:", long_lines, "label lines have more than 5 numbers (polygon labels, not plain boxes)")