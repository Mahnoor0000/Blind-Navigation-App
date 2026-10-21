# Tries several BLIP generation settings on your caption test photos and compares them.
# Plain captions only (no YOLO). Same images and references as blip_caption_test.py.
#
# Run from the FYP1 folder:
#     python scripts/blip_settings_sweep.py
#
# Output:
#   test_results/sweep_captions.csv   every caption from every setting, side by side
#   test_results/sweep_summary.txt    BLEU-4, CIDEr, caption length and time per setting
#
# Takes about (number of settings) x 31 images x ~5 s  -> roughly 15-20 minutes on CPU.

print("starting... (loading libraries can take a minute)", flush=True)

import os
import re
import csv
import time
import torch
from PIL import Image, ImageOps
from transformers import BlipProcessor, BlipForConditionalGeneration
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider

# ---------------- settings ----------------
IMAGE_DIR = "TEST IMAGES FOR BLIP"
REF_CSV = os.path.join(IMAGE_DIR, "references.csv")
BLIP_MODEL = "models/blip_qwen_drop"
OUT_DIR = "test_results/sweep_qwen_drop"

# each setting: name, text prompt (or None), and generate() options
SETTINGS = [
    ("A_current",     None, dict(num_beams=3, max_new_tokens=30, repetition_penalty=1.3)),
    ("B_greedy",      None, dict(num_beams=1, max_new_tokens=30, repetition_penalty=1.3)),
    ("C_beam5_min10", None, dict(num_beams=5, min_new_tokens=10, max_new_tokens=30,
                                 repetition_penalty=1.3, no_repeat_ngram_size=3)),
    ("D_beam5_long",  None, dict(num_beams=5, min_new_tokens=12, max_new_tokens=35,
                                 repetition_penalty=1.5, length_penalty=1.3, no_repeat_ngram_size=3)),
    ("E_prompt_picture", "a picture of", dict(num_beams=3, min_new_tokens=6, max_new_tokens=30,
                                              repetition_penalty=1.3, no_repeat_ngram_size=3)),
    ("F_prompt_view",    "a view of",    dict(num_beams=3, min_new_tokens=6, max_new_tokens=30,
                                              repetition_penalty=1.3, no_repeat_ngram_size=3)),
]
# ------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print("loading BLIP on", device, "...", flush=True)
processor = BlipProcessor.from_pretrained(BLIP_MODEL)
blip = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL).to(device).eval()


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def generate(img, prompt, opts):
    if prompt:
        inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)
    else:
        inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = blip.generate(**inputs, **opts)
    text = processor.decode(out[0], skip_special_tokens=True).strip()
    return text


def score(preds, refs):
    gts = {i: [norm(r) for r in refs[i]] for i in range(len(refs))}
    res = {i: [norm(preds[i])] for i in range(len(preds))}
    bleu, _ = Bleu(4).compute_score(gts, res, verbose=0)
    cider, _ = Cider().compute_score(gts, res)
    return bleu[3], cider


with open(REF_CSV, encoding="utf-8-sig") as f:
    items = [r for r in csv.DictReader(f) if r.get("use", "yes").strip().lower() != "no"]
items = [r for r in items if os.path.exists(os.path.join(IMAGE_DIR, r["image"]))]
refs = [[x for x in (r.get("ref1", ""), r.get("ref2", "")) if x.strip()] for r in items]
images = [ImageOps.exif_transpose(Image.open(os.path.join(IMAGE_DIR, r["image"]))).convert("RGB") for r in items]
print("images:", len(images), "| settings:", len(SETTINGS), flush=True)

captions = {}
summary = []
for name, prompt, opts in SETTINGS:
    print(f"\n=== {name} ===", flush=True)
    caps, t0 = [], time.time()
    for i, img in enumerate(images):
        caps.append(generate(img, prompt, opts))
        if (i + 1) % 10 == 0:
            print(f"   {i + 1}/{len(images)}", flush=True)
    sec = (time.time() - t0) / len(images)
    b4, cider = score(caps, refs)
    words = sum(len(c.split()) for c in caps) / len(caps)
    line = f"{name:18s} BLEU-4 {b4:.3f}   CIDEr {cider:.3f}   avg words {words:4.1f}   {sec:.1f} s/image"
    print(line, flush=True)
    summary.append(line)
    captions[name] = caps

os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "sweep_captions_vizwiz.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["no", "scene", "reference"] + [s[0] for s in SETTINGS])
    for i, r in enumerate(items):
        w.writerow([r["no"], r["scene"], refs[i][0] if refs[i] else ""] + [captions[s[0]][i] for s in SETTINGS])
with open(os.path.join(OUT_DIR, "sweep_summary_vizwiz.txt"), "w") as f:
    f.write("\n".join(summary) + "\n")

print("\n---- summary ----")
print("\n".join(summary))
print("\nAll captions side by side: test_results/sweep_captions.csv")
print("Look at images 10, 15 and 17 especially (the weak ones in the baseline).")