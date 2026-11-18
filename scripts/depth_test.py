

import os
import csv
import math
import time

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from ultralytics import YOLO

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spatial import Spatial
from fusion_engine import spatial_phrase

# ---------------- settings ----------------
IMAGE_DIR = "TEST IMAGES FOR BLIP"
REF_CSV = os.path.join(IMAGE_DIR, "references.csv")    # optional: its "scene" column picks indoor/outdoor
GROUND_TRUTH = os.path.join(IMAGE_DIR, "distances.csv")  # optional: measured distances
OUT_DIR = "test_results/depth"
YOLO_MODEL = "models/yolo_26n_clean.pt"                # the 21-class YOLO26n
YOLO_CONF = 0.35
DEPTH_MODELS = {
    "indoor": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "outdoor": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
}
INDOOR_SCENES = {"indoor", "stairs", "entrance"}       # scene names from references.csv that use the indoor model
DEFAULT_DEPTH = "outdoor"                              # for images not in references.csv
HFOV_DEG = 70                                          # phone main camera field of view along the long side (typical 65-80)

CAMERA_HEIGHT = 1.3                                    # metres above the floor when the photos were taken
PHOTO_PITCH_DEG = 5                                    # camera tilt down if no person / door / car tells us

# known object heights, floor classes, sitting-person rule: see spatial.py
# ------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT_DIR, exist_ok=True)

scene_of = {}
if os.path.exists(REF_CSV):
    with open(REF_CSV, encoding="utf-8-sig") as f:
        scene_of = {r["image"]: r.get("scene", "").strip().lower() for r in csv.DictReader(f)}

truth = []
if os.path.exists(GROUND_TRUTH):
    with open(GROUND_TRUTH, encoding="utf-8-sig") as f:
        truth = [r for r in csv.DictReader(f)]
    print("measured distances found:", len(truth))

images = sorted(f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))
print("images:", len(images), flush=True)

yolo = YOLO(YOLO_MODEL)
spatial = Spatial(cam_height=CAMERA_HEIGHT, auto_pitch=False)
depth_models = {}


def get_depth_model(kind):
    if kind not in depth_models:
        print("loading", DEPTH_MODELS[kind], flush=True)
        proc = AutoImageProcessor.from_pretrained(DEPTH_MODELS[kind])
        model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODELS[kind]).to(device).eval()
        depth_models[kind] = (proc, model)
    return depth_models[kind]


def depth_map(img, kind):
    # returns an HxW array of metres, same size as the photo
    proc, model = get_depth_model(kind)
    inputs = proc(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    pred = out.predicted_depth                         # (1, h, w) in metres for the metric models
    pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=(img.height, img.width),
                                           mode="bicubic", align_corners=False)
    return pred[0, 0].clamp(min=0).cpu().numpy()


def focal_px(img):
    # from EXIF (35 mm equivalent focal length) if the photo still has it, else from HFOV_DEG
    try:
        f35 = img.getexif().get_ifd(0x8769).get(41989)   # FocalLengthIn35mmFilm
        if f35:
            return float(f35) / 36.0 * max(img.width, img.height), "exif"
    except Exception:
        pass
    # the field of view is along the LONG side of the photo (portrait photos are narrower across)
    return max(img.width, img.height) / (2 * math.tan(math.radians(HFOV_DEG / 2))), f"hfov {HFOV_DEG}"


def where(xc):
    return "on your left" if xc < 0.33 else "on your right" if xc > 0.66 else "ahead"


def spoken(m):
    # rounded, the way it would be said: "about 2 metres", "less than 1 metre"
    if m < 1:
        return "less than 1 metre"
    r = round(m * 2) / 2 if m < 4 else round(m)
    return f"about {r:g} metre" + ("" if r == 1 else "s")


for k in DEPTH_MODELS:          # load (and download) before timing, so s/image is only the model itself
    get_depth_model(k)

rows, t_depth, t_yolo = [], [], []
for fname in images:
    path = os.path.join(IMAGE_DIR, fname)
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    W, H = img.size
    kind = "indoor" if scene_of.get(fname, "") in INDOOR_SCENES else (
        "outdoor" if fname in scene_of else DEFAULT_DEPTH)

    t0 = time.time()
    det = yolo(np.array(img)[:, :, ::-1], conf=YOLO_CONF, verbose=False)[0]
    t_yolo.append(time.time() - t0)
    t0 = time.time()
    depth = depth_map(img, kind)
    t_depth.append(time.time() - t0)
    f_px, f_src = focal_px(img)
    spatial.hfov_deg = math.degrees(2 * math.atan(max(W, H) / (2 * f_px)))   # same focal as box height

    canvas = np.array(img)[:, :, ::-1].copy()
    print(f"\n{fname}  [{kind} depth model, {t_depth[-1]:.2f} s]")
    boxes = []
    for b in det.boxes:
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        label = yolo.names[int(b.cls)]
        # depth model: median of the middle half of the box (avoids background at the box edges)
        cx1, cx2 = x1 + (x2 - x1) // 4, x2 - (x2 - x1) // 4
        cy1, cy2 = y1 + (y2 - y1) // 4, y2 - (y2 - y1) // 4
        patch = depth[cy1:max(cy2, cy1 + 1), cx1:max(cx2, cx1 + 1)]
        d_model = float(np.median(patch)) if patch.size else float("nan")
        boxes.append((x1, y1, x2, y2, label, float(b.conf), d_model))

    # pass 1: box-height distances -> this photo's camera tilt and depth-map scale
    spatial.pitch_deg = PHOTO_PITCH_DEG
    first = [spatial.measure(x1, y1, x2, y2, lab, W, H) for x1, y1, x2, y2, lab, c, dm in boxes]
    pitches = [m["pitch_obs"] for m, bx in zip(first, boxes) if m["pitch_obs"] is not None and bx[5] >= 0.5]
    pitch = float(np.median(pitches)) if pitches else PHOTO_PITCH_DEG
    ratios = [m["d_box"] / bx[6] for m, bx in zip(first, boxes) if m["d_box"] and bx[5] >= 0.5 and bx[6] > 0]
    scale = float(np.median(ratios)) if ratios else None

    # pass 2: all cues with this photo's tilt; fused = median of box height, ground contact, scaled depth
    spatial.pitch_deg = pitch
    img_rows = []
    for (x1, y1, x2, y2, label, conf, d_model) in boxes:
        d_scaled = d_model * scale if scale else None
        m = spatial.measure(x1, y1, x2, y2, label, W, H, extra=(d_scaled,) if d_scaled else ())
        r = {"image": fname, "label": label, "conf": round(conf, 2),
             "side": m["side"], "in_path": m["in_path"], "clock": m["clock"],
             "angle_deg": round(m["angle"], 1),
             "depth_model_m": round(d_model, 2),
             "box_height_m": round(m["d_box"], 2) if m["d_box"] else "",
             "ground_m": round(m["d_ground"], 2) if m["d_ground"] else "",
             "depth_scaled_m": round(d_scaled, 2) if d_scaled else "",
             "fused_m": round(m["distance"], 2) if m["distance"] else "",
             "rough": m["rough"], "cues": m["n_cues"],
             "box_h_px": y2 - y1, "pitch_deg": round(pitch, 1), "scale": round(scale, 2) if scale else "",
             "depth_model": kind, "focal": f_src}
        img_rows.append(r)
        say = spatial_phrase({**m, "label": label, "distance_smooth": m["distance"], "ttc": None})
        cues = " | ".join(f"{k} {r[k + '_m']:.1f}" for k in ("box_height", "ground", "depth_scaled")
                          if r[k + "_m"] != "")
        print(f"  {label:16s} {m['side']:14s} fused {r['fused_m'] or '-':>5} m  ({cues})  -> \"{say}\"")
        colour = (0, 0, 255) if m["in_path"] else (0, 220, 0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 3)
        txt = f"{label} {m['distance']:.1f}m" + ("?" if m["rough"] else "") if m["distance"] else label
        cv2.putText(canvas, txt, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, max(0.6, W / 1400), colour, 2)
    print(f"  (tilt {pitch:.1f} deg from {len(pitches)} objects" +
          (f", depth map rescaled x{scale:.2f})" if scale else ", depth map not rescaled)"))
    rows += img_rows

    # depth map in colour (near = warm, far = dark), next to the annotated photo
    vis = depth.copy()
    vis = (255 * (1 - (vis - vis.min()) / max(1e-6, vis.max() - vis.min()))).astype(np.uint8)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
    both = np.hstack([canvas, vis])
    scale = 1600 / both.shape[1]
    both = cv2.resize(both, (1600, int(both.shape[0] * scale)))
    cv2.imwrite(os.path.join(OUT_DIR, os.path.splitext(fname)[0] + "_depth.jpg"), both)

with open(os.path.join(OUT_DIR, "distances.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["image"])
    w.writeheader()
    w.writerows(rows)

lines = [f"images: {len(images)} | objects: {len(rows)} | YOLO {YOLO_MODEL} conf {YOLO_CONF}",
         f"time per image on {device}: YOLO {np.mean(t_yolo):.2f} s, depth model {np.mean(t_depth):.2f} s"]

# error against measured distances (largest box of that label in that photo)
if truth:
    errs = {"depth model": [], "box height": [], "ground contact": [], "depth scaled": [], "FUSED": []}
    for t in truth:
        cand = [r for r in rows if r["image"] == t["image"] and r["label"] == t["label"]]
        if not cand:
            print("not detected:", t["image"], t["label"])
            continue
        r = max(cand, key=lambda r: r["box_h_px"])
        true_m = float(t["true_m"])
        errs["depth model"].append(abs(r["depth_model_m"] - true_m) / true_m)
        if r["box_height_m"] != "":
            errs["box height"].append(abs(r["box_height_m"] - true_m) / true_m)
        for key, col in (("ground contact", "ground_m"), ("depth scaled", "depth_scaled_m"), ("FUSED", "fused_m")):
            if r[col] != "":
                errs[key].append(abs(r[col] - true_m) / true_m)
    lines.append(f"\nmeasured objects: {len(truth)}")
    for k, e in errs.items():
        if e:
            lines.append(f"  {k:12s}: mean error {100 * np.mean(e):.0f}%  median {100 * np.median(e):.0f}%"
                         f"  within 25%: {sum(x <= 0.25 for x in e)}/{len(e)}")

text = "\n".join(lines)
print("\n" + text)
with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
    f.write(text + "\n")
print(f"\nsaved {OUT_DIR}/ (images with depth maps, distances.csv, summary.txt)")