# Tests BLIP captioning on your own photos and scores it against your reference captions.
#
# For every image marked use=yes in references.csv it makes two captions per model:
#   plain    - BLIP on its own
#   grounded - YOLO labels given to BLIP as a prompt ("a photo of a car and a person")
# Then it scores both with BLEU-4 and CIDEr against ref1 (and ref2 if you filled it in),
# and writes a CSV with columns for you to judge each caption by hand.
#
# Every model in BLIP_MODELS is run with the SAME decoding settings and cleanup as the app
# (GEN_OPTIONS and clean_caption from caption.py), so the scores are directly comparable.
#
# Setup (once):   pip install pycocoevalcap
# Run from the FYP1 folder:
#     python scripts/blip_caption_test.py

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
from caption import GEN_OPTIONS, clean_caption   # same settings + cleanup the app uses

# ---------------- settings ----------------
IMAGE_DIR = "TEST IMAGES FOR BLIP"                     # folder with the photos and references.csv
REF_CSV = os.path.join(IMAGE_DIR, "references.csv")
BLIP_MODELS = {                                        # run name -> model folder
    "blip_vizwiz": "models/blip_vizwiz",
    "blip_nav_v3": "models/blip_nav_v3",
}
YOLO_MODEL = "models/best.pt"                          # run4
YOLO_CONF = 0.35
MAX_LABELS = 3
OUT_DIR = "test_results"
# ------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print("libraries loaded. device:", device, "| decoding:", GEN_OPTIONS, flush=True)
if not os.path.exists(REF_CSV):
    raise SystemExit("references.csv not found at: " + os.path.abspath(REF_CSV))
yolo = YOLO(YOLO_MODEL)


def caption(blip, processor, img, prompt=None):
    if prompt:
        inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)
    else:
        inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = blip.generate(**inputs, **GEN_OPTIONS)
    text = processor.decode(out[0], skip_special_tokens=True).strip()
    if prompt and text.lower().startswith(prompt.lower()):
        text = text[len(prompt):].strip()
    return clean_caption(text)


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
    res = {i: [norm(preds[i]) or "empty"] for i in range(len(preds))}
    bleu, _ = Bleu(4).compute_score(gts, res, verbose=0)
    cider, _ = Cider().compute_score(gts, res)
    return bleu[3], cider


def avg_words(caps):
    return sum(len(c.split()) for c in caps) / len(caps)


# ---------- read references and run YOLO once ----------
with open(REF_CSV, encoding="utf-8-sig") as f:
    refs_rows = [r for r in csv.DictReader(f) if r.get("use", "yes").strip().lower() != "no"]

items = []   # (row, image, yolo labels, refs)
for r in refs_rows:
    path = os.path.join(IMAGE_DIR, r["image"])
    if not os.path.exists(path):
        print("missing, skipped:", r["image"])
        continue
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")   # phone photos: fix rotation
    refs = [x for x in (r.get("ref1", ""), r.get("ref2", "")) if x.strip()]
    items.append((r, img, yolo_labels(img), refs))
print("images to test:", len(items))
refs_all = [it[3] for it in items]

summary = [f"images: {len(items)}", f"decoding: {GEN_OPTIONS} + clean_caption", ""]
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- caption with each model ----------
for run_name, model_path in BLIP_MODELS.items():
    print(f"\n===== {run_name} ({model_path}) =====", flush=True)
    processor = BlipProcessor.from_pretrained(model_path)
    blip = BlipForConditionalGeneration.from_pretrained(model_path).to(device).eval()

    rows, plain_all, grounded_all, times = [], [], [], []
    for r, img, labels, refs in items:
        t = time.time()
        plain = caption(blip, processor, img)
        times.append(time.time() - t)
        grounded = caption(blip, processor, img, "a photo of " + " and ".join(labels)) if labels else plain
        plain_all.append(plain)
        grounded_all.append(grounded)

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

    out_csv = os.path.join(OUT_DIR, f"captions_{run_name}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("saved", out_csv)

    pb, pc = score(plain_all, refs_all)
    gb, gc = score(grounded_all, refs_all)
    summary += [
        f"model: {run_name} ({model_path})",
        f"  plain    : BLEU-4 {pb:.3f}   CIDEr {pc:.3f}   avg words {avg_words(plain_all):.1f}",
        f"  grounded : BLEU-4 {gb:.3f}   CIDEr {gc:.3f}   avg words {avg_words(grounded_all):.1f}",
        f"  average plain caption time: {sum(times) / len(times):.2f} s per image on {device}",
        "",
    ]
    del blip, processor

# ---------- automatic scores ----------
print("\n" + "\n".join(summary))
with open(os.path.join(OUT_DIR, "captions_compare_summary.txt"), "w") as f:
    f.write("\n".join(summary))
print("saved", os.path.join(OUT_DIR, "captions_compare_summary.txt"),
      "- open the captions_*.csv files in Excel to fill in the judgement columns")