

import os
import re
import csv
import time

import torch
from PIL import Image, ImageOps
from transformers import AutoProcessor, Florence2ForConditionalGeneration, CLIPModel, CLIPProcessor
from bert_score import BERTScorer

# ---------------- settings ----------------
IMAGE_DIR = "TEST IMAGES FOR BLIP"
REF_CSV = os.path.join(IMAGE_DIR, "references.csv")     # uses ref1 and ref2 (ref2 may be empty)
OUT_DIR = "test_results"
MODEL_ID = "florence-community/Florence-2-base"         # or "florence-community/Florence-2-large" (3x bigger)
TASKS = [                                               # (name in the table, Florence task)
    ("Florence-2-base caption", "<CAPTION>"),
    #("Florence-2-base detailed", "<DETAILED_CAPTION>"),
    # ("Florence-2-base more detailed", "<MORE_DETAILED_CAPTION>"),   # a paragraph: too long to speak
]
NUM_BEAMS = 1
MAX_NEW_TOKENS = 50
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
print(f"{len(items)} photos with references", flush=True)

# 1. Florence-2 captions
print("loading", MODEL_ID, "...", flush=True)
proc = AutoProcessor.from_pretrained(MODEL_ID)
model = Florence2ForConditionalGeneration.from_pretrained(MODEL_ID).to(device).eval()
LEAD = re.compile(r"^(the|this|in this) (image|picture|photo),? (shows|is of|is|depicts|we can see)\s+", re.I)

captions, speed = {}, {}
for name, task in TASKS:
    print("captioning:", name, flush=True)
    caps, t0 = [], time.time()
    for k, img in enumerate(images, 1):
        inputs = proc(text=task, images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, num_beams=NUM_BEAMS)
        raw = proc.batch_decode(ids, skip_special_tokens=False)[0]
        text = proc.post_process_generation(raw, task=task, image_size=img.size)[task]
        text = LEAD.sub("", " ".join(text.split())).strip().rstrip(".")
        caps.append(text[:1].lower() + text[1:])
        if k == 1:
            print("  first caption:", caps[0], flush=True)
    speed[name] = (time.time() - t0) / len(images)
    captions[name] = caps
    print(f"  {speed[name]:.1f} s per image", flush=True)
del model

# 2. BERTScore (same settings as compare_captions.py)
print("BERTScore ...", flush=True)
scorer = BERTScorer(lang="en", rescale_with_baseline=True)
bert = {name: [t.tolist() for t in scorer.score(caps, refs)] for name, caps in captions.items()}

# 3. CLIPScore = 2.5 * max(cos(image, caption), 0)
clip = {}
if USE_CLIPSCORE:
    print("CLIPScore ...", flush=True)
    cm = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    cp = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    with torch.no_grad():
        for name, caps in captions.items():
            out = cm(**cp(text=caps, images=images, return_tensors="pt", padding=True,
                          truncation=True).to(device))
            iv = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            tv = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
            clip[name] = (2.5 * (iv * tv).sum(-1).clamp(min=0)).tolist()

# 4. save
os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "florence_captions.csv"), "w", newline="", encoding="utf-8-sig") as f:
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


lines = [f"{len(items)} photos | {MODEL_ID}, zero-shot, beams {NUM_BEAMS} | BERTScore roberta-large, rescaled",
         "",
         f"{'model':30s} {'BERT-P':>7} {'BERT-R':>7} {'BERT-F1':>8} {'CLIP':>6} {'words':>6} {'s/img':>6}"]
for name in captions:
    P, R, F = bert[name]
    c = f"{mean(clip[name]):6.3f}" if clip else "     -"
    words = mean([len(x.split()) for x in captions[name]])
    lines.append(f"{name:30s} {mean(P):7.3f} {mean(R):7.3f} {mean(F):8.3f} {c} {words:6.1f} {speed[name]:6.1f}")

scenes = sorted({r.get("scene", "") for r in items})
lines += ["", "BERT-F1 by scene: " + " | ".join(f"{s} ({sum(r.get('scene') == s for r in items)})" for s in scenes)]
for name in captions:
    per = [f"{mean([bert[name][2][i] for i, r in enumerate(items) if r.get('scene') == s]):.3f}" for s in scenes]
    lines.append(f"{name:30s} " + "  ".join(per))

text = "\n".join(lines)
print("\n" + text)
with open(os.path.join(OUT_DIR, "florence_summary.txt"), "w") as f:
    f.write(text + "\n")
print("\nsaved test_results/florence_captions.csv and florence_summary.txt")