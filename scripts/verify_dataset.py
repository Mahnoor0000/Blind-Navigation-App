"""
Sanity-check a YOLO-format dataset by drawing its labels back onto a
random sample of images. Run this after any conversion/merge/remap step,
BEFORE spending hours training on data you haven't actually looked at.

Usage:
    python scripts/verify_dataset.py --data path/to/data.yaml --split train --n 12
"""

import argparse
import os
import random
import yaml
from PIL import Image, ImageDraw, ImageFont


def load_data_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def draw_boxes(image_path: str, label_path: str, class_names: dict, out_path: str):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    if not os.path.exists(label_path):
        draw.text((10, 10), "NO LABEL FILE FOUND", fill="red")
        img.save(out_path)
        return 0

    n_boxes = 0
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls_id = int(parts[0])
            xc, yc, bw, bh = [float(v) for v in parts[1:5]]

            x1 = (xc - bw / 2) * w
            y1 = (yc - bh / 2) * h
            x2 = (xc + bw / 2) * w
            y2 = (yc + bh / 2) * h

            label = class_names.get(cls_id, f"UNKNOWN_ID_{cls_id}")
            draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
            draw.text((x1 + 2, max(0, y1 - 12)), label, fill="lime")
            n_boxes += 1

    img.save(out_path)
    return n_boxes


def main(data_yaml: str, split: str, n: int, out_dir: str):
    cfg = load_data_yaml(data_yaml)
    base_path = cfg.get("path", os.path.dirname(data_yaml))
    images_dir = os.path.join(base_path, cfg[split])
    labels_dir = images_dir.replace("images", "labels")
    class_names = cfg["names"]
    if isinstance(class_names, list):
        class_names = {i: name for i, name in enumerate(class_names)}

    all_images = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not all_images:
        print(f"No images found in {images_dir}")
        return

    sample = random.sample(all_images, min(n, len(all_images)))
    os.makedirs(out_dir, exist_ok=True)

    print(f"Checking {len(sample)} images from {images_dir}\n")
    total_empty = 0
    for fname in sample:
        image_path = os.path.join(images_dir, fname)
        label_path = os.path.join(labels_dir, os.path.splitext(fname)[0] + ".txt")
        out_path = os.path.join(out_dir, fname)
        n_boxes = draw_boxes(image_path, label_path, class_names, out_path)
        flag = "  <-- EMPTY, check this" if n_boxes == 0 else ""
        print(f"  {fname}: {n_boxes} box(es){flag}")
        if n_boxes == 0:
            total_empty += 1

    print(f"\nSaved annotated previews to: {out_dir}")
    print(f"{total_empty}/{len(sample)} sampled images had zero boxes — open the folder and check these look right.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, default=12, help="Number of random images to check")
    parser.add_argument("--out_dir", default="dataset_preview")
    args = parser.parse_args()

    main(args.data, args.split, args.n, args.out_dir)