import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

DA_IDS = {
    "da_indoor": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "da_outdoor": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
}
MOGE_ID = "Ruicheng/moge-2-vits-normal"


# ----------------------------------------------------------------------------- depth models
class DepthAnythingMetric:
    def __init__(self, repo_id, device, size=518):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.torch, self.device = torch, device
        self.proc = AutoImageProcessor.from_pretrained(repo_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(repo_id).to(device).eval()
        self.size = size

    def __call__(self, rgb):
        torch = self.torch
        h, w = rgb.shape[:2]
        inputs = self.proc(images=rgb, return_tensors="pt",
                           size={"height": self.size, "width": self.size})
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            pred = self.model(**inputs).predicted_depth          # (1, h', w') metres
        pred = torch.nn.functional.interpolate(pred[:, None], size=(h, w), mode="bilinear",
                                               align_corners=False)[0, 0]
        return pred.cpu().numpy().astype(np.float32), None


class MoGe2:
    def __init__(self, device, num_tokens=None):
        import torch
        from moge.model.v2 import MoGeModel
        self.torch, self.device = torch, device
        self.model = MoGeModel.from_pretrained(MOGE_ID).to(device).eval()
        self.num_tokens = num_tokens

    def __call__(self, rgb):
        torch = self.torch
        t = torch.tensor(rgb / 255.0, dtype=torch.float32, device=self.device).permute(2, 0, 1)
        kw = {"use_fp16": self.device == "cuda"}
        if self.num_tokens:
            kw["num_tokens"] = self.num_tokens
        with torch.no_grad():
            out = self.model.infer(t, **kw)
        depth = out["depth"].cpu().numpy().astype(np.float32)     # metres, inf where masked (sky)
        fx = float(out["intrinsics"][0, 0])
        if fx > 10:                                                # pixel units -> normalise
            fx /= rgb.shape[1]
        hfov = math.degrees(2 * math.atan(0.5 / fx))               # MoGe's estimate of camera FOV
        return depth, hfov


# ----------------------------------------------------------------------------- helpers
def box_depth(depth, box):
    """median depth of the lower-middle part of the box (avoids background at edges/top)"""
    x1, y1, x2, y2 = map(int, box)
    bw, bh = x2 - x1, y2 - y1
    patch = depth[y1 + int(0.5 * bh): y2, x1 + int(0.25 * bw): x2 - int(0.25 * bw)]
    patch = patch[np.isfinite(patch) & (patch > 0)]
    return float(np.median(patch)) if patch.size > 10 else np.nan


def pick_box(dets, cls, px=None, py=None):
    cands = [d for d in dets if d[0] == cls]
    if not cands:
        return None
    if px is not None and py is not None:
        inside = [d for d in cands if d[1][0] <= px <= d[1][2] and d[1][1] <= py <= d[1][3]]
        if inside:   # smallest box containing the point
            return min(inside, key=lambda d: (d[1][2] - d[1][0]) * (d[1][3] - d[1][1]))
    return max(cands, key=lambda d: (d[1][2] - d[1][0]) * (d[1][3] - d[1][1]))  # largest = usually target


def num_or_none(v):
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def save_vis(path, rgb, maps, rows):
    tiles = [cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)]
    for name, d in maps.items():
        v = d.copy()
        v[~np.isfinite(v)] = np.nanmax(v[np.isfinite(v)]) if np.isfinite(v).any() else 0
        v = cv2.applyColorMap(cv2.normalize(-np.log(v + 0.1), None, 0, 255, cv2.NORM_MINMAX)
                              .astype(np.uint8), cv2.COLORMAP_TURBO)
        cv2.putText(v, name, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
        tiles.append(v)
    for t in tiles:
        for r in rows:
            if r.get("box"):
                x1, y1, x2, y2 = map(int, r["box"])
                cv2.rectangle(t, (x1, y1), (x2, y2), (255, 255, 255), 3)
                cv2.putText(t, f"{r['class']} true {r['true_m']}m", (x1, max(30, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    h = 360
    tiles = [cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h)) for t in tiles]
    cv2.imwrite(str(path), np.hstack(tiles))


def summarise(df, methods, times):
    lines = []
    bins = [0, 2, 4, 8, 1e9]
    labels = ["<2m", "2-4m", "4-8m", ">8m"]
    df = df.copy()
    df["bin"] = pd.cut(df["true_m"], bins=bins, labels=labels, right=False)
    hdr = f"{'method':<14}{'n':>4}{'MAE m':>8}{'mean %':>8}{'med %':>8}{'bias %':>8}" + \
          "".join(f"{l:>9}" for l in labels) + f"{'indoor%':>9}{'outdr%':>9}{'s/img':>8}"
    lines += ["Error vs tape measure  (% columns = mean absolute % error; bias<0 = underestimates)",
              hdr, "-" * len(hdr)]
    for m in methods:
        if m not in df:
            continue
        sub = df[np.isfinite(df[m])]
        if sub.empty:
            lines.append(f"{m:<14}   0  (no valid estimates)")
            continue
        err = sub[m] - sub["true_m"]
        pct = 100 * err.abs() / sub["true_m"]
        row = f"{m:<14}{len(sub):>4}{err.abs().mean():>8.2f}{pct.mean():>8.1f}{pct.median():>8.1f}" \
              f"{(100 * err / sub['true_m']).mean():>8.1f}"
        for l in labels:
            p = pct[sub["bin"] == l]
            row += f"{p.mean():>9.1f}" if len(p) else f"{'-':>9}"
        for sc in ["indoor", "outdoor"]:
            p = pct[sub["scene"].str.lower() == sc]
            row += f"{p.mean():>9.1f}" if len(p) else f"{'-':>9}"
        t = times.get(m)
        row += f"{t:>8.2f}" if t else f"{'~0':>8}"
        lines.append(row)
    missed = df["detected"].eq(False).sum()
    lines += ["", f"objects not detected by YOLO (excluded): {missed}"]
    if "moge_hfov" in df and df["moge_hfov"].notna().any():
        lines.append(f"MoGe-estimated horizontal FOV: median {df['moge_hfov'].median():.1f} deg "
                     f"(compare with your measured --hfov)")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- no ground truth
def run_without_gt(args, yolo, models, DistanceEstimator, out_dir):
    """Every detection in every image: distance from each method side by side, plus how much
    the methods agree. No truth, so this shows behaviour and agreement, not accuracy."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in exts)
    if not paths:
        raise SystemExit(f"no images in {args.images}")
    rows, times = [], {m: [] for m in models}
    for path in paths:
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        res = yolo.predict(bgr, conf=args.conf, verbose=False)[0]
        maps, hfov = {}, None
        for m, fn in models.items():
            t0 = time.perf_counter()
            maps[m], h = fn(rgb)
            times[m].append(time.perf_counter() - t0)
            hfov = h if m == "moge" else hfov
        est = DistanceEstimator(W, H, hfov_deg=args.hfov, cam_height_m=args.cam_height) \
            if DistanceEstimator else None
        img_rows = []
        print(f"\n{path.name}" + (f"  (MoGe FOV estimate {hfov:.0f} deg)" if hfov else ""))
        for b, c, s in zip(res.boxes.xyxy.cpu().tolist(), res.boxes.cls.cpu().tolist(),
                           res.boxes.conf.cpu().tolist()):
            cls = res.names[int(c)]
            r = {"image": path.name, "class": cls, "conf": round(s, 2), "box": [round(v) for v in b],
                 "true_m": np.nan, "detected": True}
            for m, d in maps.items():
                r[m] = box_depth(d, b)
            if est:
                gc, ph = est.ground_contact(cls, b), est.pinhole(cls, b)
                r["ground"], r["pinhole"] = (gc or np.nan), (ph or np.nan)
                f = est.estimate(cls, b, depth_map=maps.get("moge") if "moge" in maps
                                 else next(iter(maps.values()), None))
                r["fused"], r["direction"], r["spoken"] = (f["distance_m"] or np.nan), f["direction"], f["spoken"]
            vals = [r[k] for k in [*maps, "ground", "pinhole"] if k in r and np.isfinite(r[k])]
            r["spread_%"] = 100 * (max(vals) - min(vals)) / np.median(vals) if len(vals) > 1 else np.nan
            line = "  ".join(f"{k} {r[k]:.2f}" for k in [*maps, "ground", "pinhole", "fused"]
                             if k in r and np.isfinite(r[k]))
            print(f"  {cls:<16} {line}  | spread {r['spread_%']:.0f}%  | \"{r.get('spoken', '')}\"")
            rows.append(r)
            img_rows.append(r)
        if args.save_vis:
            save_vis_nogt(out_dir / f"vis_{path.stem}.jpg", rgb, maps, img_rows)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "estimates_no_gt.csv", index=False)
    lines = ["", "NO GROUND TRUTH -> agreement between methods, not accuracy",
             f"{'method':<12}{'n':>5}{'median ratio to MoGe':>22}{'s/img':>8}"]
    ref = "moge" if "moge" in df else None
    for m in [*models, "ground", "pinhole", "fused"]:
        if m not in df:
            continue
        s = df[np.isfinite(df[m])]
        ratio = (s[m] / s[ref]).median() if ref and m != ref and len(s) else np.nan
        t = np.mean(times[m]) if m in times and times[m] else 0
        lines.append(f"{m:<12}{len(s):>5}{ratio:>22.2f}{t:>8.2f}")
    if "spread_%" in df:
        lines.append(f"\nmedian disagreement between methods: {df['spread_%'].median():.0f}% "
                     f"(objects with >40%: {(df['spread_%'] > 40).sum()} - look at those images)")
    print("\n".join(lines))
    (out_dir / "summary_no_gt.txt").write_text("\n".join(lines))


def save_vis_nogt(path, rgb, maps, rows):
    tiles = [cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)]
    names = ["image"] + list(maps)
    for d in maps.values():
        v = d.copy()
        fin = np.isfinite(v)
        v[~fin] = v[fin].max() if fin.any() else 0
        tiles.append(cv2.applyColorMap(cv2.normalize(-np.log(v + 0.1), None, 0, 255, cv2.NORM_MINMAX)
                                       .astype(np.uint8), cv2.COLORMAP_TURBO))
    for t, n in zip(tiles, names):
        cv2.putText(t, n, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
        for r in rows:
            x1, y1, x2, y2 = map(int, r["box"])
            key = "fused" if n == "image" else n
            val = r.get(key, np.nan)
            cv2.rectangle(t, (x1, y1), (x2, y2), (255, 255, 255), 2)
            txt = f"{r['class']} {val:.1f}m" if np.isfinite(val) else r["class"]
            cv2.putText(t, txt, (x1 + 4, y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    h = 360
    tiles = [cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h)) for t in tiles]
    cv2.imwrite(str(path), np.hstack(tiles))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--gt", default=None, help="ground-truth CSV; omit to just compare methods")
    ap.add_argument("--yolo", required=True)
    ap.add_argument("--models", default="da_indoor,da_outdoor,moge")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--hfov", type=float, default=66.0, help="your camera's horizontal FOV (deg)")
    ap.add_argument("--cam-height", type=float, default=1.3, help="camera height above floor (m)")
    ap.add_argument("--da-size", type=int, default=518)
    ap.add_argument("--moge-tokens", type=int, default=None, help="e.g. 1200 for faster CPU runs")
    ap.add_argument("--out", default="depth_results")
    ap.add_argument("--save-vis", action="store_true")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(args.yolo)
    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    models = {}
    for m in wanted:
        print(f"loading {m} ...")
        models[m] = MoGe2(device, args.moge_tokens) if m == "moge" else \
            DepthAnythingMetric(DA_IDS[m], device, args.da_size)

    try:
        from distance_fusion import DistanceEstimator
    except ImportError:
        DistanceEstimator = None
        print("distance_fusion.py not found -> geometry cues skipped")

    if args.gt is None:
        return run_without_gt(args, yolo, models, DistanceEstimator, out_dir)

    gt = pd.read_csv(args.gt)
    gt.columns = [c.strip().lower() for c in gt.columns]
    for c in ["px", "py"]:
        if c not in gt:
            gt[c] = np.nan

    times = {m: [] for m in models}
    rows_out = []
    for img_name, group in gt.groupby("image", sort=False):
        path = Path(args.images) / img_name
        bgr = cv2.imread(str(path))
        if bgr is None:
            print(f"!! cannot read {path}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]

        res = yolo.predict(bgr, conf=args.conf, verbose=False)[0]
        dets = [(res.names[int(c)], b.tolist(), float(s))
                for b, c, s in zip(res.boxes.xyxy.cpu(), res.boxes.cls.cpu(), res.boxes.conf.cpu())]

        maps, moge_hfov = {}, None
        for m, fn in models.items():
            t0 = time.perf_counter()
            d, hfov = fn(rgb)
            times[m].append(time.perf_counter() - t0)
            maps[m] = d
            if m == "moge":
                moge_hfov = hfov

        est = DistanceEstimator(W, H, hfov_deg=args.hfov, cam_height_m=args.cam_height) \
            if DistanceEstimator else None

        vis_rows = []
        for _, g in group.iterrows():
            cls = str(g["class"]).strip()
            r = {"image": img_name, "class": cls, "true_m": float(g["true_m"]),
                 "scene": str(g.get("scene", "")).strip().lower()}
            det = pick_box(dets, cls, num_or_none(g["px"]), num_or_none(g["py"]))
            r["detected"] = det is not None
            if det:
                box = det[1]
                r["box"], r["yolo_conf"] = [round(v) for v in box], round(det[2], 3)
                for m, d in maps.items():
                    r[m] = box_depth(d, box)
                if "da_indoor" in r and "da_outdoor" in r:     # the checkpoint you'd actually use
                    r["da_matched"] = r["da_indoor"] if r["scene"] == "indoor" else r["da_outdoor"]
                if est:
                    gc, ph = est.ground_contact(cls, box), est.pinhole(cls, box)
                    r["ground"] = gc if gc else np.nan
                    r["pinhole"] = ph if ph else np.nan
                    depth_for_fusion = maps.get("moge")
                    fused = est.estimate(cls, box, depth_map=depth_for_fusion)["distance_m"]
                    r["fused_moge"] = fused if fused else np.nan
            r["moge_hfov"] = moge_hfov
            rows_out.append(r)
            vis_rows.append(r)
            shown = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()
                     if k in ["class", "true_m", *models, "da_matched", "ground", "pinhole", "fused_moge"]}
            print(img_name, shown)

        if args.save_vis:
            save_vis(out_dir / f"vis_{Path(img_name).stem}.jpg", rgb, maps, vis_rows)

    df = pd.DataFrame(rows_out)
    df.to_csv(out_dir / "results.csv", index=False)
    mean_t = {m: float(np.mean(v)) for m, v in times.items() if v}
    mean_t["da_matched"] = mean_t.get("da_indoor")
    if "fused_moge" in df:
        mean_t["fused_moge"] = mean_t.get("moge")
    methods = ["da_indoor", "da_outdoor", "da_matched", "moge", "ground", "pinhole", "fused_moge"]
    report = summarise(df, methods, mean_t) + f"\n\ndevice: {device}, da_size {args.da_size}, " \
                                              f"moge_tokens {args.moge_tokens or 'default'}"
    print("\n" + report)
    (out_dir / "summary.txt").write_text(report)
    print(f"\nsaved {out_dir / 'results.csv'} and summary.txt")


if __name__ == "__main__":
    main()