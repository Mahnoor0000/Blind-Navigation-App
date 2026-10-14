"""
Convert YOLO segmentation-format labels (class_id + polygon points) to
YOLO detection-format labels (class_id + bbox), for training a detector
(not a segmentation model) like plain YOLOv8n.

Segmentation line:  class_id x1 y1 x2 y2 x3 y3 ... (variable length)
Detection line:     class_id x_center y_center width height (always 5 numbers)

Logic: for each polygon, take min/max x and min/max y across all its
points -> that's the tightest bounding box around the shape.

Verified: tested against two hand-built squares with known coordinates
(0.1,0.1)-(0.3,0.3) and (0.5,0.4)-(0.7,0.6) -- output bbox center/size
matched the expected values exactly.

Usage:
    python scripts/convert_seg_to_bbox.py --labels_dir path/to/labels/train --out_dir path/to/bbox_labels/train
"""

import argparse
import os


def polygon_line_to_bbox_line(line: str) -> str:
    parts = line.strip().split()
    cls_id = parts[0]
    coords = [float(v) for v in parts[1:]]

    xs = coords[0::2]  # every even index = x
    ys = coords[1::2]  # every odd index = y

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    return f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def convert_file(in_path: str, out_path: str) -> int:
    out_lines = []
    with open(in_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out_lines.append(polygon_line_to_bbox_line(line))

    with open(out_path, "w") as f:
        f.write("\n".join(out_lines))
        if out_lines:
            f.write("\n")

    return len(out_lines)


def convert_directory(labels_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    total_files, total_boxes = 0, 0

    for fname in os.listdir(labels_dir):
        if not fname.endswith(".txt"):
            continue
        in_path = os.path.join(labels_dir, fname)
        out_path = os.path.join(out_dir, fname)
        n_boxes = convert_file(in_path, out_path)
        total_files += 1
        total_boxes += n_boxes

    print(f"Converted {total_files} label files, {total_boxes} total boxes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    convert_directory(args.labels_dir, args.out_dir)