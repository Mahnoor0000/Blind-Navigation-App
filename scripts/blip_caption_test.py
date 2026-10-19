# Tests BLIP captioning on your own photos and scores it against your reference captions.
#
# For every image marked use=yes in references.csv it makes two captions:
#   plain    - BLIP on its own
#   grounded - YOLO labels given to BLIP as a prompt ("a photo of a car and a person")
# Then it scores both with BLEU-4 and CIDEr against ref1 (and ref2 if you filled it in),
# and writes a CSV with columns for you to judge each caption by hand.
#
# Setup (once):   pip install pycocoevalcap
# Run from the FYP1 folder:
#     python scripts/blip_caption_test.py
#
# Later, for the fine-tuned model: set BLIP_MODEL to the downloaded "best" folder,
# change RUN_NAME, run again on the same images, and compare the two summaries.

print("starting... (loading torch, YOLO and BLIP libraries can take a minute)", flush=True)

import os
import re
import csv
import time
import torch
from PIL import Image, ImageOps
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider

# ---------------- settings ----------------
IMAGE_DIR = "TEST IMAGES FOR BLIP"                     # folder with the photos and references.csv
REF_CSV = os.path.join(IMAGE_DIR, "references.csv")
BLIP_MODEL = "Salesforce/blip-image-captioning-base"   # later: path to the fine-tuned "best" folder
RUN_NAME = "blip_base"                                 # used in the output file names
YOLO_MODEL = "models/best.pt"                          # run4
YOLO_CONF = 0.35
MAX_LABELS = 3
OUT_DIR = "test_results"
# ------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print("libraries loaded. loading models on", device, "(first run downloads BLIP, about 1 GB)...", flush=True)
if not os.path.exists(REF_CSV):
    raise SystemExit("references.csv not found at: " + os.path.abspath(REF_CSV))
processor = BlipProcessor.from_pretrained(BLIP_MODEL)
blip = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL).to(device).eval()
yolo = YOLO(YOLO_MODEL)


def caption(img, prompt=None):
    if prompt:
        inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)
    else:
        inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = blip.generate(**inputs, num_beams=3, max_new_tokens=30, repetition_penalty=1.3)
    return processor.decode(out[0], skip_special_tokens=True).strip()


def yolo_labels(img):
    r = yolo.predict(img, conf=YOLO_CONF, verbose=False)[0]
    counts = {}
    for c in r.boxes.cls.tolist():
        name = yolo.names[int(c)].replace("_", " ")
        counts[name] = counts.get(name, 0) + 1
    return [n for n, _ in sorted(counts.items(), key=lambda kv: -kv[1])][:MAX_LABELS]


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def score(preds, refs):
    gts = {i: [norm(r) for r in refs[i]] for i in range(len(refs))}
    res = {i: [norm(preds[i])] for i in range(len(preds))}
    bleu, _ = Bleu(4).compute_score(gts, res, verbose=0)
    cider, _ = Cider().compute_score(gts, res)
    return bleu[3], cider


# ---------- read references ----------
with open(REF_CSV, encoding="utf-8-sig") as f:
    refs_rows = [r for r in csv.DictReader(f) if r.get("use", "yes").strip().lower() != "no"]
print("images to test:", len(refs_rows))

rows, plain_all, grounded_all, refs_all, times = [], [], [], [], []
for r in refs_rows:
    path = os.path.join(IMAGE_DIR, r["image"])
    if not os.path.exists(path):
        print("missing, skipped:", r["image"])
        continue
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")   # phone photos: fix rotation
    labels = yolo_labels(img)

    t = time.time()
    plain = caption(img)
    times.append(time.time() - t)
    grounded = caption(img, "a photo of " + " and ".join(labels)) if labels else plain

    refs = [x for x in (r.get("ref1", ""), r.get("ref2", "")) if x.strip()]
    plain_all.append(plain)
    grounded_all.append(grounded)
    refs_all.append(refs)

    print(f"{r['no']:>3}  [{r['scene']}]")
    print("     yolo    :", ", ".join(labels) if labels else "-")
    print("     plain   :", plain)
    print("     grounded:", grounded)
    print("     ref     :", refs[0] if refs else "-")

    rows.append({
        "no": r["no"], "image": r["image"], "scene": r["scene"],
        "yolo labels": ", ".join(labels), "reference": refs[0] if refs else "",
        "plain caption": plain, "grounded caption": grounded,
        "plain correct? (y/n)": "", "plain invents objects? (y/n)": "",
        "grounded correct? (y/n)": "", "grounded invents objects? (y/n)": "",
        "better for walking (plain/grounded/neither)": "",
    })

os.makedirs(OUT_DIR, exist_ok=True)
out_csv = os.path.join(OUT_DIR, f"captions_{RUN_NAME}.csv")
with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# ---------- automatic scores ----------
pb, pc = score(plain_all, refs_all)
gb, gc = score(grounded_all, refs_all)
summary = [
    f"model: {BLIP_MODEL}",
    f"images: {len(rows)}",
    f"plain    : BLEU-4 {pb:.3f}   CIDEr {pc:.3f}   avg words {sum(len(c.split()) for c in plain_all) / len(rows):.1f}",
    f"grounded : BLEU-4 {gb:.3f}   CIDEr {gc:.3f}   avg words {sum(len(c.split()) for c in grounded_all) / len(rows):.1f}",
    f"average plain caption time: {sum(times) / len(times):.2f} s per image on {device}",
]
print("\n" + "\n".join(summary))
with open(os.path.join(OUT_DIR, f"captions_{RUN_NAME}_summary.txt"), "w") as f:
    f.write("\n".join(summary) + "\n")
print("\nsaved", out_csv, "- open it in Excel and fill in the judgement columns")