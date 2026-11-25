"""
Class-count bar chart for a YOLO dataset (IEEE single-column size).

Usage (Colab or local):
    python class_counts_chart.py --data /content/dataset/data.yaml --splits train
    python class_counts_chart.py --data data.yaml --splits train val test --log

It reads your data.yaml for the class names and the split folders, then counts
how many boxes (instances) of each class are in the YOLO label .txt files.
Outputs: class_counts.pdf (for LaTeX), class_counts.png, class_counts.csv
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


def label_dir_for(images_path: Path) -> Path:
    """Ultralytics convention: .../images/... -> .../labels/..."""
    parts = list(images_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)
    return images_path  # labels stored next to images


def count_split(label_dir: Path, n_classes: int):
    boxes, images = Counter(), Counter()
    files = list(label_dir.rglob("*.txt"))
    for f in files:
        seen = set()
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            c = int(float(line.split()[0]))
            if 0 <= c < n_classes:
                boxes[c] += 1
                seen.add(c)
        for c in seen:
            images[c] += 1
    return boxes, images, len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"C:\Users\hp\Desktop\FYP1\merged_dataset_v3\data.yaml", help="path to data.yaml")
    ap.add_argument("--splits", nargs="+", default=["train"])
    ap.add_argument("--no-log", dest="log", action="store_false", help="linear x axis")
    ap.add_argument("--out", default="class_counts")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.data))
    names = cfg["names"]
    if isinstance(names, dict):  # {0: 'person', ...}
        names = [names[k] for k in sorted(names)]
    root = Path(cfg.get("path", Path(args.data).parent))

    per_split = {}
    for s in args.splits:
        if s not in cfg:
            cfg[s] = f"images/{s}"
        img_path = Path(cfg[s])
        if not img_path.is_absolute():
            img_path = root / img_path
        lbl = label_dir_for(img_path)
        if not lbl.exists():
            # data.yaml paths are from another machine (e.g. Colab):
            # fall back to <folder of data.yaml>/labels/<split>
            here = Path(args.data).resolve().parent
            for cand in (here / "labels" / s, here / "labels" / ("val" if s == "valid" else s)):
                if cand.exists():
                    print(f"note: {lbl} not found, using {cand}")
                    lbl = cand
                    break
            else:
                print(f"WARNING: no label folder found for split '{s}' (tried {lbl} and {here / 'labels' / s})")
        boxes, imgs, n_files = count_split(lbl, len(names))
        per_split[s] = boxes
        print(f"\n{s}: {n_files} label files, {sum(boxes.values())} boxes")
        for i in sorted(range(len(names)), key=lambda i: -boxes[i]):
            print(f"  {names[i]:<20}{boxes[i]:>8} boxes{imgs[i]:>8} images")

    # CSV
    with open(f"{args.out}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class"] + args.splits)
        for i, n in enumerate(names):
            w.writerow([n] + [per_split[s][i] for s in args.splits])

    # Plot: horizontal bars, sorted by count of the first split
    first = args.splits[0]
    order = sorted(range(len(names)), key=lambda i: per_split[first][i])
    labels = [names[i].replace("_", " ") for i in order]

    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, ax = plt.subplots(figsize=(3.5, 0.17 * len(names) + 0.8))
    k = len(args.splits)
    h = 0.8 / k
    colors = ["#2f6db5", "#9a9a9a", "#c0504d", "#e8a33d"]
    for j, s in enumerate(args.splits):
        y = [p + (j - (k - 1) / 2) * h for p in range(len(order))]
        ax.barh(y, [per_split[s][i] for i in order], height=h,
                color=colors[j % len(colors)], label=s)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Number of instances (boxes)")
    if args.log:
        ax.set_xscale("log")
    if k > 1:
        ax.legend(frameon=False, loc="lower right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{args.out}.pdf")
    fig.savefig(f"{args.out}.png", dpi=300)
    print(f"\nSaved {args.out}.pdf, {args.out}.png, {args.out}.csv")


if __name__ == "__main__":
    main()