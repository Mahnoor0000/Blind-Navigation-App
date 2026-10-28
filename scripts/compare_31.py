"""
Compare blip_vizwiz vs blip_nav_v3 on your own test photos (runs on laptop CPU).

Usage (from your FYP1 folder):
    python compare_31.py "TEST IMAGES FOR BLIP"

Writes to test_results/:
    blind_rating_31.csv   rate this first (A/B order is shuffled)
    blind_key_31.csv      open only AFTER rating
    timing_31.txt         seconds per caption for each model
"""
import os, sys, csv, time, random, glob
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

PHOTO_DIR = sys.argv[1] if len(sys.argv) > 1 else "TEST IMAGES FOR BLIP"
MODELS = {
    "blip_vizwiz": "models/blip_vizwiz",
    "blip_nav_v3": "models/blip_nav_v3",
}
OUT_DIR = "test_results"
SEED = 42

# Same decoding for both models: setting A (3 beams) + repetition guard
GEN = dict(num_beams=3, max_length=45, no_repeat_ngram_size=3)

torch.set_num_threads(max(1, os.cpu_count() or 1))
photos = sorted(p for p in glob.glob(os.path.join(PHOTO_DIR, "*"))
                if p.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))
assert photos, f"No images found in {PHOTO_DIR}"
print(f"{len(photos)} photos")

captions, timing = {}, {}
for name, path in MODELS.items():
    print(f"\nLoading {name} from {path}")
    proc = BlipProcessor.from_pretrained(path)
    model = BlipForConditionalGeneration.from_pretrained(path).eval()
    caps, t0 = [], time.time()
    for i, p in enumerate(photos, 1):
        img = Image.open(p).convert("RGB")
        with torch.no_grad():
            out = model.generate(**proc(images=img, return_tensors="pt"), **GEN)
        caps.append(proc.decode(out[0], skip_special_tokens=True).strip())
        print(f"  {i}/{len(photos)} {os.path.basename(p)}: {caps[-1]}")
    timing[name] = (time.time() - t0) / len(photos)
    captions[name] = caps
    del model

os.makedirs(OUT_DIR, exist_ok=True)
rng = random.Random(SEED)
with open(f"{OUT_DIR}/blind_rating_31.csv", "w", newline="", encoding="utf-8") as f, \
     open(f"{OUT_DIR}/blind_key_31.csv", "w", newline="", encoding="utf-8") as k:
    w, wk = csv.writer(f), csv.writer(k)
    w.writerow(["photo", "caption_A", "caption_B", "A_correct(y/n)", "B_correct(y/n)",
                "A_invents(y/n)", "B_invents(y/n)", "better_for_walking(A/B/same/neither)"])
    wk.writerow(["photo", "A_is", "B_is"])
    for i, p in enumerate(photos):
        a, b = captions["blip_vizwiz"][i], captions["blip_nav_v3"][i]
        flip = rng.random() < 0.5
        w.writerow([os.path.basename(p), b if flip else a, a if flip else b, "", "", "", "", ""])
        wk.writerow([os.path.basename(p),
                     "blip_nav_v3" if flip else "blip_vizwiz",
                     "blip_vizwiz" if flip else "blip_nav_v3"])

with open(f"{OUT_DIR}/timing_31.txt", "w") as f:
    for name, s in timing.items():
        line = f"{name}: {s:.2f} s per caption (CPU)"
        print(line); f.write(line + "\n")

print(f"\nDone. Rate {OUT_DIR}/blind_rating_31.csv BEFORE opening blind_key_31.csv.")