import os
import random
import shutil
import xml.etree.ElementTree as ET
from PIL import Image, ImageOps

# folder where you extracted the kaggle stairs dataset (images + xml files)
stairs_dir = r"D:\Downloads\archive"

# the merged 20 class dataset made by add_extra_sources.py
out_dir = "Dataset/merged_dataset_v2"

stairs_id = 5          # id of "stairs" in the 20 class list
max_side = 960         # phone photos are 3000+ px, shrink them to save disk and time
train_part = 0.85
valid_part = 0.10

random.seed(42)

# step 1: find every xml file and the image that goes with it
images = {}
xmls = []
for root, folders, files in os.walk(stairs_dir):
    for fname in files:
        low = fname.lower()
        full = os.path.join(root, fname)
        if low.endswith((".jpg", ".jpeg", ".png")):
            images[os.path.splitext(fname)[0]] = full
        elif low.endswith(".xml"):
            xmls.append(full)

print("images found:", len(images), " xml files found:", len(xmls))

random.shuffle(xmls)

added = {"train": 0, "valid": 0, "test": 0}
no_image = 0
no_stairs = 0
size_mismatch = 0
other_names = set()

for n in range(len(xmls)):
    xml_path = xmls[n]
    stem = os.path.splitext(os.path.basename(xml_path))[0]

    if stem not in images:
        no_image += 1
        continue

    tree = ET.parse(xml_path)
    root = tree.getroot()
    xml_w = int(root.find("size/width").text)
    xml_h = int(root.find("size/height").text)

    # open the image the same way yolo will (phone photos carry a rotation tag)
    img = Image.open(images[stem])
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")

    if img.size != (xml_w, xml_h):
        size_mismatch += 1
        continue

    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        if name != "stairs":
            other_names.add(name)
            continue
        box = obj.find("bndbox")
        x1 = float(box.find("xmin").text)
        y1 = float(box.find("ymin").text)
        x2 = float(box.find("xmax").text)
        y2 = float(box.find("ymax").text)

        # keep the box inside the image
        x1 = max(0, min(x1, xml_w))
        x2 = max(0, min(x2, xml_w))
        y1 = max(0, min(y1, xml_h))
        y2 = max(0, min(y2, xml_h))
        if x2 <= x1 or y2 <= y1:
            continue

        x_center = (x1 + x2) / 2 / xml_w
        y_center = (y1 + y2) / 2 / xml_h
        width = (x2 - x1) / xml_w
        height = (y2 - y1) / xml_h
        lines.append(str(stairs_id) + " " + str(round(x_center, 6)) + " " + str(round(y_center, 6)) + " " + str(round(width, 6)) + " " + str(round(height, 6)))

    if len(lines) == 0:
        no_stairs += 1
        continue

    position = n / len(xmls)
    if position < train_part:
        split = "train"
    elif position < train_part + valid_part:
        split = "valid"
    else:
        split = "test"

    # shrink big images, the normalised boxes stay correct
    img.thumbnail((max_side, max_side))

    new_stem = "stairs_" + stem
    img.save(os.path.join(out_dir, "images", split, new_stem + ".jpg"), quality=92)
    with open(os.path.join(out_dir, "labels", split, new_stem + ".txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    added[split] += 1

print("added:", added)
print("skipped, no matching image:", no_image)
print("skipped, no stairs box:", no_stairs)
print("skipped, image size does not match xml:", size_mismatch)
if len(other_names) > 0:
    print("other class names seen and ignored:", other_names)