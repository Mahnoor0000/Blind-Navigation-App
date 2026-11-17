# Compares caption models on the caption test photos with BERTScore (and CLIPScore).
#
#   BERTScore : how close the MEANING of a caption is to your reference caption(s)
#               (uses word embeddings, so "footpath" ~ "walkway" counts, unlike BLEU / CIDEr)
#   CLIPScore : how well the caption matches the PHOTO itself (no reference needed)
#
# Run from the FYP1 folder:
#   pip install bert-score
#   python scripts/compare_captions.py
#
# First run downloads roberta-large (~1.4 GB, for BERTScore) and CLIP (~600 MB).
#
# Outputs:
#   test_results/compare_captions.csv   every caption + its scores, model by model
#   test_results/compare_summary.txt    one table for the report

import os
import csv
import time

import torch
from PIL import Image, ImageOps
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPModel, CLIPProcessor
from bert_score import BERTScorer

# ---------------- settings ----------------
IMAGE_DIR = "TEST IMAGES FOR BLIP"
REF_CSV = os.path.join(IMAGE_DIR, "references.csv")     # uses ref1 and ref2 (ref2 may be empty)
OUT_DIR = "test_results"

MODELS = [                                              # (name in the table, model folder or HF name)
    ("BLIP-base (COCO)", "Salesforce/blip-image-captioning-base"),
    ("BLIP + VizWiz", "models/blip_vizwiz"),
    ("BLIP + VizWiz + Qwen", "models/blip_nav_v3"),
]
# same decoding as the app (caption.py)
GEN = dict(num_beams=3, max_length=45, no_repeat_ngram_size=3, repetition_penalty=1.3)
USE_CLIPSCORE = True
# ------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

with open(REF_CSV, encoding="utf-8-sig") as f:
    items = [r for r in csv.DictReader(f) if r.get("use", "yes").strip().lower() != "no"]
items = [r for r in items if os.path.exists(os.path.join(IMAGE_DIR, r["image"]))]
refs = [[x.strip() for x in (r.get("ref1", ""), r.get("ref2", "")) if x and x.strip()] for r in items]
keep = [i for i, rf in enumerate(refs) if rf]
items, refs = [items[i] for i in keep], [refs[i] for i in keep]
images = [ImageOps.exif_transpose(Image.open(os.path.join(IMAGE_DIR, r["image"]))).convert("RGB") for r in items]
n_ref2 = sum(len(rf) > 1 for rf in refs)
print(f"{len(items)} photos with references ({n_ref2} also have ref2)", flush=True)

# 1. captions from every model
captions, speed = {}, {}
for name, path in MODELS:
    if not path.startswith("Salesforce/") and not os.path.isdir(path):
        print("skipped (folder not found):", name, path)
        continue
    print("captioning with", name, "...", flush=True)
    proc = BlipProcessor.from_pretrained(path)
    model = BlipForConditionalGeneration.from_pretrained(path).to(device).eval()
    caps, t0 = [], time.time()
    for img in images:
        with torch.no_grad():
            out = model.generate(**proc(images=img, return_tensors="pt").to(device), **GEN)
        caps.append(proc.decode(out[0], skip_special_tokens=True).strip().rstrip("."))
    speed[name] = (time.time() - t0) / len(images)
    captions[name] = caps
    del model

# 2. BERTScore (rescaled with baseline: 0 ~ unrelated sentence, 1 = same meaning)
print("BERTScore ...", flush=True)
scorer = BERTScorer(lang="en", rescale_with_baseline=True)
bert = {}
for name, caps in captions.items():
    P, R, F = scorer.score(caps, refs)          # several references per photo: best match counts
    bert[name] = (P.tolist(), R.tolist(), F.tolist())

# 3. CLIPScore (Hessel et al. 2021): 2.5 * max(cos(image, caption), 0)
clip = {}
if USE_CLIPSCORE:
    print("CLIPScore ...", flush=True)
    cm = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    cp = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    with torch.no_grad():
        for name, caps in captions.items():
            # full forward pass: image_embeds / text_embeds come back projected and normalised
            # (works on old and new transformers versions)
            out = cm(**cp(text=caps, images=images, return_tensors="pt", padding=True,
                          truncation=True).to(device))
            iv, tv = out.image_embeds, out.text_embeds
            iv = iv / iv.norm(dim=-1, keepdim=True)
            tv = tv / tv.norm(dim=-1, keepdim=True)
            clip[name] = (2.5 * (iv * tv).sum(-1).clamp(min=0)).tolist()

# 4. save
os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "compare_captions.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    head = ["no", "scene", "ref1", "ref2"]
    for name in captions:
        head += [name, name + " BERT-F1"] + ([name + " CLIP"] if clip else [])
    w.writerow(head)
    for i, r in enumerate(items):
        row = [r["no"], r.get("scene", ""), refs[i][0], refs[i][1] if len(refs[i]) > 1 else ""]
        for name in captions:
            row += [captions[name][i], round(bert[name][2][i], 3)] + ([round(clip[name][i], 3)] if clip else [])
        w.writerow(row)


def mean(x):
    return sum(x) / len(x)


lines = [f"{len(items)} photos | BERTScore roberta-large, rescaled | references per photo: 1-2",
         "",
         f"{'model':24s} {'BERT-P':>7} {'BERT-R':>7} {'BERT-F1':>8} {'CLIP':>6} {'words':>6} {'s/img':>6}"]
for name in captions:
    P, R, F = bert[name]
    c = f"{mean(clip[name]):6.3f}" if clip else "     -"
    words = mean([len(x.split()) for x in captions[name]])
    lines.append(f"{name:24s} {mean(P):7.3f} {mean(R):7.3f} {mean(F):8.3f} {c} {words:6.1f} {speed[name]:6.1f}")

# BERT-F1 per scene type (indoor / street / stairs ...)
scenes = sorted({r.get("scene", "") for r in items})
lines += ["", "BERT-F1 by scene: " + " | ".join(f"{s} ({sum(r.get('scene') == s for r in items)})" for s in scenes)]
for name in captions:
    per = []
    for s in scenes:
        idx = [i for i, r in enumerate(items) if r.get("scene") == s]
        per.append(f"{mean([bert[name][2][i] for i in idx]):.3f}")
    lines.append(f"{name:24s} " + "  ".join(per))

text = "\n".join(lines)
print("\n" + text)
with open(os.path.join(OUT_DIR, "compare_summary.txt"), "w") as f:
    f.write(text + "\n")
print("\nsaved test_results/compare_captions.csv and compare_summary.txt")