import argparse
import os
import shutil
import xml.etree.ElementTree as ET


def find_pairs(split_dir: str):
    """Returns list of (jpg_path, xml_path) for every matched pair in split_dir."""
    pairs = []
    for fname in os.listdir(split_dir):
        if fname.lower().endswith(".xml"):
            stem = fname[:-4]
            jpg_path = os.path.join(split_dir, stem + ".jpg")
            xml_path = os.path.join(split_dir, fname)
            if os.path.exists(jpg_path):
                pairs.append((jpg_path, xml_path))
            else:
                print(f"  WARNING: no matching .jpg for {fname}, skipping")
    return pairs


def collect_class_names(dataset_root: str, splits: list) -> list:
    """First pass: scan every XML across all splits, return sorted unique class names."""
    classes = set()
    for split in splits:
        split_dir = os.path.join(dataset_root, split)
        if not os.path.isdir(split_dir):
            continue
        for fname in os.listdir(split_dir):
            if not fname.lower().endswith(".xml"):
                continue
            tree = ET.parse(os.path.join(split_dir, fname))
            for obj in tree.getroot().findall("object"):
                classes.add(obj.find("name").text)
    return sorted(classes)


def convert_xml_to_yolo_lines(xml_path: str, class_to_id: dict) -> list:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    w = int(root.find("size/width").text)
    h = int(root.find("size/height").text)

    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in class_to_id:
            continue  # shouldn't happen since we collected classes first, but safe
        cls_id = class_to_id[name]

        b = obj.find("bndbox")
        xmin = float(b.find("xmin").text)
        ymin = float(b.find("ymin").text)
        xmax = float(b.find("xmax").text)
        ymax = float(b.find("ymax").text)

        xc = ((xmin + xmax) / 2) / w
        yc = ((ymin + ymax) / 2) / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h

        if bw <= 0 or bh <= 0:
            continue

        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    return lines


def convert_split(dataset_root: str, out_dir: str, split: str, class_to_id: dict):
    split_dir = os.path.join(dataset_root, split)
    if not os.path.isdir(split_dir):
        print(f"Skipping '{split}' -- folder not found at {split_dir}")
        return 0, 0

    out_images_dir = os.path.join(out_dir, "images", split)
    out_labels_dir = os.path.join(out_dir, "labels", split)
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)

    pairs = find_pairs(split_dir)
    total_images, total_boxes = 0, 0

    for jpg_path, xml_path in pairs:
        stem = os.path.splitext(os.path.basename(jpg_path))[0]
        lines = convert_xml_to_yolo_lines(xml_path, class_to_id)

        out_label_path = os.path.join(out_labels_dir, stem + ".txt")
        with open(out_label_path, "w") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")

        out_image_path = os.path.join(out_images_dir, stem + ".jpg")
        shutil.copy2(jpg_path, out_image_path)

        total_images += 1
        total_boxes += len(lines)

    print(f"  {split}: {total_images} images, {total_boxes} boxes")
    return total_images, total_boxes


def write_data_yaml(out_dir: str, class_names: list, has_test: bool):
    lines = [
        f"path: {os.path.abspath(out_dir)}",
        "train: images/train",
        "val: images/valid",
    ]
    if has_test:
        lines.append("test: images/test")
    lines.append("names:")
    for i, name in enumerate(class_names):
        lines.append(f"  {i}: {name}")

    with open(os.path.join(out_dir, "data.yaml"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main(dataset_root: str, out_dir: str):
    splits = ["train", "valid", "test"]

    print("Scanning all splits for class names...")
    class_names = collect_class_names(dataset_root, splits)
    class_to_id = {name: i for i, name in enumerate(class_names)}
    print(f"Found {len(class_names)} classes: {class_names}\n")

    print("Converting splits:")
    has_test = os.path.isdir(os.path.join(dataset_root, "test"))
    for split in splits:
        convert_split(dataset_root, out_dir, split, class_to_id)

    write_data_yaml(out_dir, class_names, has_test)
    print(f"\nDone. data.yaml written to {out_dir}/data.yaml")
    print("NEXT STEP: run verify_dataset.py against it before training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True, help="Folder containing train/valid/test subfolders")
    parser.add_argument("--out_dir", required=True, help="Where to write the converted YOLO-format dataset")
    args = parser.parse_args()

    main(args.dataset_root, args.out_dir)