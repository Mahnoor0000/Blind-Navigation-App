# Auto-labels UNLABELLED stairs photos with Grounding DINO (finds objects from a text prompt),
# so you only have to delete the bad ones instead of drawing boxes yourself.
# (YOLO-World returned zero boxes on your install, so this uses Grounding DINO via transformers.)
#
# Run from the FYP1 folder:
#
#   STEP 0  MODE = "check"
#       python scripts/autolabel_stairs.py
#       -> tries 5 images, prints the boxes found and the seconds per image. Nothing is saved.
#
#   STEP 1  MODE = "label"
#       -> writes Dataset/stairs_autolabel/images + labels (YOLO format, class 0 = stairs)
#       -> writes a small preview of every kept image to Dataset/stairs_autolabel/review/
#          (images without a confident, sensibly sized box are skipped automatically)
#
#   REVIEW  open the review folder in Explorer (View > Extra large icons), Ctrl+click every
#           preview where the box is wrong, and delete them together.
#
#   STEP 2  MODE = "finalize"
#       -> removes every image whose preview you deleted
#
# Then merge Dataset/stairs_autolabel with add_pothole_data.py
# (SRC_DIR = "Dataset/stairs_autolabel", PREFIX = "stairs3_", SRC_KEEP = {0: 5}).

import os
import time
import random
import hashlib

import cv2
import torch
import torchvision
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# ---------------- settings ----------------
SRC_DIR = r"D:\Downloads\new stair data\stairs_normal_dataset\stairs"   # unlabelled stairs photos
OUT_DIR = "Dataset/stairs_autolabel"
MODE = "finalize"                                # "check" -> "label" -> review -> "finalize"

GD_MODEL = "IDEA-Research/grounding-dino-tiny"   # downloads automatically (~700 MB) the first time
TEXT = "a staircase. stairs. steps."             # lowercase, each phrase ends with a period
BOX_THRESHOLD = 0.35                             # only confident boxes are kept
TEXT_THRESHOLD = 0.25
MIN_BOX_AREA = 0.03                           # ignore tiny boxes (< 3% of the image)
MAX_BOX_AREA = 0.90                           # ignore boxes that cover basically the whole image
MAX_BOXES = 3                                 # images with more boxes are usually messy, skip them
MAX_IMAGES = 600                              # random sample; ~16 s each on your CPU = ~2.5 hours
MAX_SIDE = 800                                # shrink photos first: faster, and plenty for labels

# images you test with must never be trained on
TEST_DIRS = ["test_images_descending", "test_images"]
# ------------------------------------------

IMG_TYPES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
img_out = os.path.join(OUT_DIR, "images")
lbl_out = os.path.join(OUT_DIR, "labels")
review = os.path.join(OUT_DIR, "review")


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_image(path):
    im = cv2.imread(path)
    if im is None:
        return None
    H, W = im.shape[:2]
    if max(H, W) > MAX_SIDE:
        s = MAX_SIDE / max(H, W)
        im = cv2.resize(im, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
    return im


def detect(im_bgr):
    """returns a list of (x1, y1, x2, y2, score) in pixels, overlapping boxes merged"""
    pil = Image.fromarray(cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB))
    inputs = processor(images=pil, text=TEXT, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs)
    size = [pil.size[::-1]]
    try:   # newer transformers
        res = processor.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD, target_sizes=size)[0]
    except TypeError:   # older transformers
        res = processor.post_process_grounded_object_detection(
            out, inputs.input_ids, box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD, target_sizes=size)[0]
    boxes, scores = res["boxes"].cpu(), res["scores"].cpu()
    if len(boxes) == 0:
        return []
    keep = torchvision.ops.nms(boxes, scores, 0.5)   # the 3 phrases often box the same stairs
    return [(*boxes[k].tolist(), float(scores[k])) for k in keep]


def sensible(dets, W, H):
    return [d for d in dets if MIN_BOX_AREA <= (d[2] - d[0]) * (d[3] - d[1]) / (W * H) <= MAX_BOX_AREA]


# ---------- finalize ----------
if MODE == "finalize":
    kept = {os.path.splitext(f)[0] for f in os.listdir(review)}
    removed = 0
    for f in os.listdir(img_out):
        stem = os.path.splitext(f)[0]
        if stem not in kept:
            os.remove(os.path.join(img_out, f))
            lbl = os.path.join(lbl_out, stem + ".txt")
            if os.path.exists(lbl):
                os.remove(lbl)
            removed += 1
    left = len(os.listdir(img_out))
    boxes = sum(len(open(os.path.join(lbl_out, f)).read().split("\n")) - 1
                for f in os.listdir(lbl_out))
    print("removed", removed, "images whose preview you deleted")
    print("kept", left, "images with", boxes, "stairs boxes")
    print("next: merge", OUT_DIR, "with add_pothole_data.py (PREFIX stairs3_, SRC_KEEP {0: 5})")
    raise SystemExit

# ---------- load model and list images ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("loading", GD_MODEL, "on", DEVICE, "(first time downloads it)...")
processor = AutoProcessor.from_pretrained(GD_MODEL)
model = AutoModelForZeroShotObjectDetection.from_pretrained(GD_MODEL).to(DEVICE).eval()

images = []
for root, _, files in os.walk(SRC_DIR):
    for f in files:
        if f.lower().endswith(IMG_TYPES):
            images.append(os.path.join(root, f))
images.sort()
print("images found:", len(images))

# ---------- check ----------
if MODE == "check":
    for path in images[:5]:
        im = load_image(path)
        t = time.time()
        dets = detect(im)
        sec = time.time() - t
        H, W = im.shape[:2]
        ok = sensible(dets, W, H)
        print(os.path.basename(path), "->", len(dets), "boxes, scores", [round(d[4], 2) for d in dets],
              "| kept after size filter:", len(ok), f"| {sec:.1f} s")
    print("\nIf most images got a box, set MODE = 'label'.")
    print("Time for", min(MAX_IMAGES, len(images)), "images is roughly that many x the seconds above.")
    raise SystemExit

# ---------- label ----------
if os.path.exists(OUT_DIR):
    raise SystemExit(OUT_DIR + " already exists. Delete it to start again, or set MODE = 'finalize'.")
for d in (img_out, lbl_out, review):
    os.makedirs(d)

test_hashes = set()
for d in TEST_DIRS:
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.lower().endswith(IMG_TYPES):
                test_hashes.add(md5(os.path.join(d, f)))
print("test images to exclude:", len(test_hashes))

if len(images) > MAX_IMAGES:
    random.Random(0).shuffle(images)
    images = sorted(images[:MAX_IMAGES])
    print("using a random sample of", len(images))

seen = set()
n_boxes = n_empty = n_test = n_dup = 0
start = time.time()
for i, path in enumerate(images):
    h = md5(path)
    if h in test_hashes:
        n_test += 1
        continue
    if h in seen:
        n_dup += 1
        continue
    seen.add(h)

    im = load_image(path)
    if im is None:
        continue
    H, W = im.shape[:2]
    dets = sensible(detect(im), W, H)

    # automatic filter: no confident box, or too many boxes -> skip the image entirely
    if not dets or len(dets) > MAX_BOXES:
        n_empty += 1
    else:
        name = f"img_{i:05d}"
        lines = []
        preview = im.copy()
        for x1, y1, x2, y2, s in dets:
            lines.append(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                         f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
            cv2.rectangle(preview, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 3)
            cv2.putText(preview, f"stairs {s:.2f}", (int(x1), max(20, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
        cv2.imwrite(os.path.join(img_out, name + ".jpg"), im)
        with open(os.path.join(lbl_out, name + ".txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        n_boxes += len(lines)
        s = 400 / max(preview.shape[:2])
        preview = cv2.resize(preview, (int(preview.shape[1] * s), int(preview.shape[0] * s)))
        cv2.imwrite(os.path.join(review, name + ".jpg"), preview)

    if (i + 1) % 25 == 0:
        per = (time.time() - start) / (i + 1)
        print(f"  {i + 1}/{len(images)} done, about {per * (len(images) - i - 1) / 60:.0f} min left")

print()
print("skipped, same as a test image:", n_test)
print("skipped, exact duplicates:", n_dup)
print("checked images:", len(seen))
print("auto-skipped (no confident box or too many boxes):", n_empty)
print("kept for review:", len(os.listdir(review)), "| stairs boxes:", n_boxes)
print("NOW: open", review, "in Explorer with View > Extra large icons,")
print("     Ctrl+click every wrong one and delete them together.")
print("Then set MODE = 'finalize' and run again.")