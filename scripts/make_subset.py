import os
import random
import shutil
import yaml

src_dir = "Dataset/merged_dataset_v3"
out_dir = "Dataset/merged_subset"

# how many images in each split
targets = {"train": 10000, "valid": 600, "test": 300}
per_class = {"train": 800, "valid": 30, "test": 15}

# the 91 phone photos of stairs from the first stairs dataset, always keep them in train
always_keep_prefix = "stairs_"

random.seed(42)

if os.path.exists(out_dir):
    print("output folder already exists, delete it first:", out_dir)
    raise SystemExit

with open(os.path.join(src_dir, "data.yaml")) as f:
    info = yaml.safe_load(f)
names = info["names"]

for split in ["train", "valid", "test"]:
    img_dir = os.path.join(src_dir, "images", split)
    lbl_dir = os.path.join(src_dir, "labels", split)

    # image file for every stem
    image_files = {}
    for fname in os.listdir(img_dir):
        image_files[os.path.splitext(fname)[0]] = fname

    # which classes are in each label file
    class_to_stems = {}
    all_stems = []
    for fname in os.listdir(lbl_dir):
        if not fname.endswith(".txt"):
            continue
        stem = fname[:-4]
        if stem not in image_files:
            continue
        all_stems.append(stem)
        found = set()
        with open(os.path.join(lbl_dir, fname)) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 5:
                    found.add(int(parts[0]))
        for c in found:
            if c not in class_to_stems:
                class_to_stems[c] = []
            class_to_stems[c].append(stem)

    chosen = set()

    if split == "train":
        for stem in all_stems:
            if stem.startswith(always_keep_prefix):
                chosen.add(stem)

    # rarest classes first, so they are not crowded out
    order = sorted(class_to_stems.keys(), key=lambda c: len(class_to_stems[c]))
    for c in order:
        pool = class_to_stems[c][:]
        random.shuffle(pool)
        have = 0
        for stem in pool:
            if stem in chosen:
                have += 1
        for stem in pool:
            if have >= per_class[split]:
                break
            if stem not in chosen:
                chosen.add(stem)
                have += 1

    # fill the rest with random images
    rest = [s for s in all_stems if s not in chosen]
    random.shuffle(rest)
    for stem in rest:
        if len(chosen) >= targets[split]:
            break
        chosen.add(stem)

    os.makedirs(os.path.join(out_dir, "images", split))
    os.makedirs(os.path.join(out_dir, "labels", split))

    box_counts = {}
    for stem in chosen:
        shutil.copy2(os.path.join(img_dir, image_files[stem]), os.path.join(out_dir, "images", split, image_files[stem]))
        shutil.copy2(os.path.join(lbl_dir, stem + ".txt"), os.path.join(out_dir, "labels", split, stem + ".txt"))
        with open(os.path.join(lbl_dir, stem + ".txt")) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 5:
                    c = int(parts[0])
                    box_counts[c] = box_counts.get(c, 0) + 1

    print(split, "images:", len(chosen))
    if split == "train":
        for c in range(len(names)):
            print("  ", names[c], "-", box_counts.get(c, 0))

data = {
    "path": os.path.abspath(out_dir),
    "train": "images/train",
    "val": "images/valid",
    "test": "images/test",
    "nc": len(names),
    "names": names,
}
with open(os.path.join(out_dir, "data.yaml"), "w") as f:
    yaml.safe_dump(data, f, sort_keys=False)

print("done:", out_dir)