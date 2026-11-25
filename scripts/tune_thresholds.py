# Finds a confidence threshold for every YOLO class from data, instead of by hand.
#
#   normal classes : the threshold with the best F1 (balance of false alarms and misses)
#   safety classes : the threshold with the best F2 score: like F1, but a miss counts 4x more than
#                    a false alarm (missing stairs is worse than a false warning). The earlier rule
#                    "at least 80% recall" pushed person / vehicle / stairs to the 0.15 floor, because
#                    YOLO26n never reaches 80% recall for them -> far too many false alarms.
#
# Uses the VALIDATION split, so the test split stays untouched for the final numbers.
#
# Run from the FYP1 folder:
#   python scripts/tune_thresholds.py
# Then copy the printed CLASS_CONF block into fusion_engine.py.

import numpy as np
from ultralytics import YOLO

# ---------------- settings ----------------
MODEL = "models/yolo_26n_clean.pt"
DATA = "Dataset/merged_dataset_v3/data.yaml"    # its valid split was never used for training
SPLIT = "val"
SAFETY = {"stairs", "pothole", "vehicle", "person", "obstacle", "pole"}
BETA = 2.0                                     # F2: recall weighted 2x (misses cost 4x)
MIN_T, MAX_T = 0.15, 0.80                      # never go outside this range
# ------------------------------------------

model = YOLO(MODEL)
res = model.val(data=DATA, split=SPLIT, imgsz=640, batch=8, conf=0.001, plots=False, verbose=False)
box = res.box
px = np.asarray(box.px)                        # 1000 confidence values from 0 to 1
names = model.names

rows = []
for k, c in enumerate(box.ap_class_index):
    name = names[int(c)]
    p, r, f1 = np.asarray(box.p_curve[k]), np.asarray(box.r_curve[k]), np.asarray(box.f1_curve[k])
    ok = (px >= MIN_T) & (px <= MAX_T)
    if name in SAFETY:
        fb = (1 + BETA ** 2) * p * r / np.maximum(BETA ** 2 * p + r, 1e-9)
        cand = np.where(ok)[0]
        i = cand[np.argmax(fb[cand])]
        rule = "best F2 (safety)"
    else:
        cand = np.where(ok)[0]
        i = cand[np.argmax(f1[cand])]
        rule = "best F1"
    rows.append((name, float(px[i]), float(p[i]), float(r[i]), float(f1[i]), rule))

print(f"\n{'class':18s} {'threshold':>9} {'precision':>9} {'recall':>7} {'F1':>6}  rule")
for name, t, p, r, f, rule in rows:
    print(f"{name:18s} {t:9.2f} {p:9.2f} {r:7.2f} {f:6.2f}  {rule}")

print("\n# paste into fusion_engine.py (replaces CLASS_CONF):")
print("CLASS_CONF = {")
for name, t, *_ in rows:
    print(f'    "{name}": {t:.2f},')
print("}")
print("DEFAULT_CONF = 0.50   # for any class not listed")