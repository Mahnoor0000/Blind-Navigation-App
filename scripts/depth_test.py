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

# real heights in metres, for the box-height method (only classes with a fairly fixed size)
KNOWN_HEIGHT = {
    "person": 1.65, "door": 2.0, "vehicle": 1.5, "elevator": 2.1, "fire_hydrant": 0.75,
    "traffic_cone": 0.7, "stop_sign": 0.75, "fire_extinguisher": 0.5, "bin": 0.9, "chair": 0.9,
    "bench": 0.8, "dog": 0.6,
}
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

    canvas = np.array(img)[:, :, ::-1].copy()
    print(f"\n{fname}  [{kind} depth model, {t_depth[-1]:.2f} s]")
    img_rows = []
    for b in det.boxes:
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        label = yolo.names[int(b.cls)]
        # depth model: median of the middle half of the box (avoids background at the box edges)
        cx1, cx2 = x1 + (x2 - x1) // 4, x2 - (x2 - x1) // 4
        cy1, cy2 = y1 + (y2 - y1) // 4, y2 - (y2 - y1) // 4
        patch = depth[cy1:max(cy2, cy1 + 1), cx1:max(cx2, cx1 + 1)]
        d_model = float(np.median(patch)) if patch.size else float("nan")
        # box height: only for known sizes and boxes not cut off at the top / bottom of the photo
        cut = y1 <= 2 or y2 >= H - 3
        d_box = (f_px * KNOWN_HEIGHT[label] / max(1, y2 - y1)) if label in KNOWN_HEIGHT and not cut else float("nan")
        xc = (x1 + x2) / 2 / W
        img_rows.append({"image": fname, "label": label, "conf": round(float(b.conf), 2), "position": where(xc),
                     "depth_model_m": round(d_model, 2), "box_height_m": round(d_box, 2) if d_box == d_box else "",
                     "box_h_px": y2 - y1, "depth_model": kind, "focal": f_src})
        img_rows[-1]["_box"] = (x1, y1, x2, y2)

    # hybrid: the depth model gets the ORDER right but its metre scale drifts, so rescale the whole
    # depth map with the objects whose real size we know (median of box-height / depth-model)
    ratios = [r["box_height_m"] / r["depth_model_m"] for r in img_rows
              if r["box_height_m"] != "" and r["conf"] >= 0.5 and r["depth_model_m"] > 0]
    scale = float(np.median(ratios)) if ratios else None
    for r in img_rows:
        r["depth_scaled_m"] = round(r["depth_model_m"] * scale, 2) if scale else ""
        r["scale"] = round(scale, 2) if scale else ""
        best = r["depth_scaled_m"] if scale else r["depth_model_m"]
        box_txt = f" | box {r['box_height_m']:.1f} m" if r["box_height_m"] != "" else ""
        sc_txt = f" | scaled {r['depth_scaled_m']:.1f} m" if scale else ""
        print(f"  {r['label']:18s} {r['position']:14s} depth model {r['depth_model_m']:5.1f} m{box_txt}{sc_txt}"
              f"   -> say: \"{r['label'].replace('_', ' ')} {r['position']}, {spoken(best)}\"")
        x1, y1, x2, y2 = r.pop("_box")
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 3)
        cv2.putText(canvas, f"{r['label']} {best:.1f}m", (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.6, W / 1400), (0, 220, 0), 2)
    if scale:
        print(f"  (depth map rescaled x{scale:.2f} from {len(ratios)} known-size objects)")
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
    errs = {"depth model": [], "box height": [], "depth scaled": []}
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
        if r["depth_scaled_m"] != "":
            errs["depth scaled"].append(abs(r["depth_scaled_m"] - true_m) / true_m)
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