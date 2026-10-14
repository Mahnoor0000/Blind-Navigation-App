import os
import yaml
from collections import Counter

# point this at your merged dataset (the one with images/labels folders)
dataset_dir = "Dataset/merged_dataset_v2"

# read class names from data.yaml so the list is always complete
with open(os.path.join(dataset_dir, "data.yaml")) as f:
    names = yaml.safe_load(f)["names"]
if isinstance(names, dict):              # some yaml files store names as {0: "person", ...}
    names = [names[k] for k in sorted(names)]

counts = Counter()
split_counts = {}

for split in ["train", "valid", "test"]:
    labels_dir = os.path.join(dataset_dir, "labels", split)
    if not os.path.isdir(labels_dir):
        print("skipping", split, "not found at", labels_dir)
        continue

    split_counts[split] = Counter()
    for fname in os.listdir(labels_dir):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(labels_dir, fname)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                class_id = int(line.split()[0])
                counts[class_id] += 1
                split_counts[split][class_id] += 1

print("classes in data.yaml:", len(names))
print()
print("id".ljust(4), "class".ljust(20), "train".rjust(7), "valid".rjust(7), "test".rjust(7), "total".rjust(8))
for i, name in enumerate(names):
    row = [split_counts.get(s, Counter()).get(i, 0) for s in ["train", "valid", "test"]]
    print(str(i).ljust(4), name.ljust(20), *[str(n).rjust(7) for n in row], str(counts.get(i, 0)).rjust(8))

# warn about label ids that are not in data.yaml (a sign of a broken merge)
unknown = sorted(k for k in counts if k >= len(names) or k < 0)
for k in unknown:
    print("WARNING: class id", k, "appears", counts[k], "times but is not in data.yaml")

print()
print("total boxes:", sum(counts.values()))