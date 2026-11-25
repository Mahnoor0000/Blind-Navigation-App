
import argparse
import csv
import tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def local_yaml(data_path):
    """Rewrite data.yaml so it points at the folders next to it (fixes old Colab paths)."""
    cfg = yaml.safe_load(open(data_path, encoding="utf-8"))
    root = Path(data_path).resolve().parent
    fixed = {
        "path": str(root).replace("\\", "/"),
        "train": "images/train",
        "val": "images/test",
        "test": "images/test",
        "names": cfg["names"],
    }
    tmp = Path(tempfile.gettempdir()) / "paper_eval_data.yaml"
    yaml.safe_dump(fixed, open(tmp, "w", encoding="utf-8"), allow_unicode=True)
    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return tmp, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--stairs", default=None, help="folder with the descending-stairs photos")
    args = ap.parse_args()

    model = YOLO(args.weights)
    tag = Path(args.weights).parent.parent.name or Path(args.weights).stem
    data_yaml, names = local_yaml(args.data)

    # ---------- 1. test-split metrics ----------
    print(f"\n=== {args.weights} on the test split ===")
    res = model.val(data=str(data_yaml), split="test", imgsz=640, batch=16,
                    plots=False, verbose=False)
    b = res.box
    print(f"\nALL  P {b.mp:.3f}  R {b.mr:.3f}  mAP50 {b.map50:.3f}  mAP50-95 {b.map:.3f}\n")

    rows = []
    print(f"{'class':<20}{'P':>7}{'R':>7}{'mAP50':>8}{'mAP50-95':>10}")
    for i, c in enumerate(b.ap_class_index):
        p, r, ap50, ap5095 = b.class_result(i)
        name = names[int(c)]
        rows.append([name, round(p, 3), round(r, 3), round(ap50, 3), round(ap5095, 3)])
        print(f"{name:<20}{p:>7.3f}{r:>7.3f}{ap50:>8.3f}{ap5095:>10.3f}")
    missing = [n for k, n in enumerate(names) if k not in set(int(c) for c in b.ap_class_index)]
    if missing:
        print("classes with no test boxes:", ", ".join(missing))

    out = f"paper_eval_{tag}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "P", "R", "mAP50", "mAP50-95"])
        w.writerow(["ALL", round(b.mp, 3), round(b.mr, 3), round(b.map50, 3), round(b.map, 3)])
        w.writerows(rows)
    print(f"\nsaved {out}")

    # ---------- 2. descending stairs ----------
    if args.stairs:
        if "stairs" not in names:
            print("no 'stairs' class in this model's data.yaml, skipping stairs test")
            return
        sid = names.index("stairs")
        imgs = sorted(p for p in Path(args.stairs).rglob("*") if p.suffix.lower() in IMG_EXT)
        print(f"\n=== descending stairs: {len(imgs)} images ===")
        best = []        # highest stairs confidence per image (0 if none)
        wrong = {}       # other labels the model put on these photos
        for p in imgs:
            r = model.predict(str(p), conf=0.05, imgsz=640, verbose=False)[0]
            confs = [float(c) for c, k in zip(r.boxes.conf, r.boxes.cls) if int(k) == sid]
            best.append(max(confs) if confs else 0.0)
            for c, k in zip(r.boxes.conf, r.boxes.cls):
                if int(k) != sid and float(c) >= 0.25:
                    n = model.names[int(k)]
                    wrong[n] = wrong.get(n, 0) + 1
        for t in (0.25, 0.5):
            found = sum(c >= t for c in best)
            print(f"  conf {t}: stairs found in {found}/{len(imgs)} images")
        print(f"  never found (even at 0.05): {sum(c == 0 for c in best)} images")
        if wrong:
            print("  other labels on these photos (conf >= 0.25):",
                  ", ".join(f"{k} x{v}" for k, v in sorted(wrong.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()