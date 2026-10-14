import argparse
import os
import shutil

UNIFIED_CLASSES = ["person", "vehicle", "cycle", "pole", "tree", "stairs",
                    "door", "chair", "table", "shelf", "bin", "fire_extinguisher", "obstacle"]
UNIFIED_ID = {name: i for i, name in enumerate(UNIFIED_CLASSES)}

INDOOR_MAP = {
    "human": "person", "bin": "bin", "chair": "chair", "door": "door",
    "fire extinguisher": "fire_extinguisher", "shelf": "shelf", "table": "table",
}
OUTDOOR_MAP = {
    "Person": "person", "Bin": "bin",
    "Bus": "vehicle", "CNG": "vehicle", "Car": "vehicle", "Motorcycle": "vehicle",
    "Pickup": "vehicle", "Rickshaw": "vehicle", "Truck": "vehicle", "Van": "vehicle",
    "Van gari": "vehicle", "Leguna": "vehicle", "Food Van": "vehicle",
    "Cycle": "cycle",
    "Electric Pole": "pole", "Building Pillar": "pole",
    "Tree": "tree", "Stairs": "stairs",
    "Obstacle": "obstacle", "Parking Cone": "obstacle", "Food cart": "obstacle",
}


def read_names(yaml_path):
    content = open(yaml_path).read()
    after = content.split("names:", 1)[1].strip()
    if after.startswith("["):
        return [n.strip().strip("'\"") for n in after.split("]")[0].strip("[]").split(",")]
    names = []
    for line in after.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            break
        names.append(line.split(":", 1)[1].strip())
    return names


def get_split_dirs(source_dir, split):
    a_img, a_lbl = os.path.join(source_dir, split, "images"), os.path.join(source_dir, split, "labels")
    b_img, b_lbl = os.path.join(source_dir, "images", split), os.path.join(source_dir, "labels", split)
    if os.path.isdir(a_lbl):
        return a_img, a_lbl
    if os.path.isdir(b_lbl):
        return b_img, b_lbl
    return None, None


def merge_split(src_img_dir, src_lbl_dir, out_dir, split, id_map, prefix):
    out_img_dir = os.path.join(out_dir, "images", split)
    out_lbl_dir = os.path.join(out_dir, "labels", split)
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    n_img, n_box = 0, 0
    for fname in os.listdir(src_lbl_dir):
        if not fname.endswith(".txt"):
            continue
        stem = fname[:-4]

        lines = []
        for line in open(os.path.join(src_lbl_dir, fname)):
            parts = line.split()
            if not parts:
                continue
            old_id = int(parts[0])
            if old_id not in id_map:
                continue
            parts[0] = str(id_map[old_id])
            lines.append(" ".join(parts))
            n_box += 1

        src_img = None
        for ext in (".jpg", ".jpeg", ".png"):
            if os.path.exists(os.path.join(src_img_dir, stem + ext)):
                src_img = os.path.join(src_img_dir, stem + ext)
                break
        if not src_img:
            continue

        new_stem = f"{prefix}_{stem}"
        open(os.path.join(out_lbl_dir, new_stem + ".txt"), "w").write("\n".join(lines) + ("\n" if lines else ""))
        shutil.copy2(src_img, os.path.join(out_img_dir, new_stem + os.path.splitext(src_img)[1]))
        n_img += 1

    return n_img, n_box


def merge_source(source_dir, class_map, out_dir, prefix, splits):
    names = read_names(os.path.join(source_dir, "data.yaml"))
    id_map = {i: UNIFIED_ID[class_map[n]] for i, n in enumerate(names) if n in class_map}

    for canon_split, src_split in splits.items():
        img_dir, lbl_dir = get_split_dirs(source_dir, src_split)
        if not lbl_dir:
            print(f"[{prefix}] no {src_split} split found")
            continue
        n_img, n_box = merge_split(img_dir, lbl_dir, out_dir, canon_split, id_map, prefix)
        print(f"[{prefix}] {canon_split}: {n_img} images, {n_box} boxes")


def write_data_yaml(out_dir):
    lines = [f"path: {os.path.abspath(out_dir)}", "train: images/train",
             "val: images/valid", "test: images/test", "names:"]
    lines += [f"  {i}: {n}" for i, n in enumerate(UNIFIED_CLASSES)]
    open(os.path.join(out_dir, "data.yaml"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--indoor_dir", required=True)
    parser.add_argument("--outdoor_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    merge_source(args.indoor_dir, INDOOR_MAP, args.out_dir, "indoor",
                 {"train": "train", "valid": "valid", "test": "test"})
    merge_source(args.outdoor_dir, OUTDOOR_MAP, args.out_dir, "outdoor",
                 {"train": "train", "valid": "valid", "test": "test"})
    write_data_yaml(args.out_dir)
    print("done:", os.path.join(args.out_dir, "data.yaml"))