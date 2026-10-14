import os

dataset_dir = "Dataset/merged_dataset_v2"
tree_id = 4

# boxes smaller than this (fraction of the image) are removed
min_area = 0.02
# an image with a tree box bigger than this is removed completely (forest / park shots)
max_area = 0.7
# an image with more tree boxes than this is removed (crowds of plants, forests, vines)
max_boxes = 4

# False = only count and print, nothing is changed. set to True to really change files
apply_changes = False

images_removed_big = 0
images_removed_crowd = 0
images_removed_empty = 0
tiny_boxes_removed = 0
files_changed = 0
tree_files = 0

for split in ["train", "valid", "test"]:
    lbl_dir = os.path.join(dataset_dir, "labels", split)
    img_dir = os.path.join(dataset_dir, "images", split)

    for fname in os.listdir(lbl_dir):
        # only the open images tree files, the other sources are left alone
        if not fname.startswith("tree_") or not fname.endswith(".txt"):
            continue
        tree_files += 1
        lbl_path = os.path.join(lbl_dir, fname)
        stem = fname[:-4]

        lines = []
        with open(lbl_path) as f:
            for line in f:
                if line.strip() != "":
                    lines.append(line.strip())

        has_big = False
        tree_count = 0
        keep = []
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            if int(parts[0]) == tree_id:
                tree_count += 1
                area = float(parts[3]) * float(parts[4])
                if area > max_area:
                    has_big = True
                if area < min_area:
                    tiny_boxes_removed += 1
                    continue
            keep.append(line)

        # decide what happens to this image
        remove_image = False
        if has_big:
            images_removed_big += 1
            remove_image = True
        elif tree_count > max_boxes:
            images_removed_crowd += 1
            remove_image = True
        elif len(keep) == 0:
            images_removed_empty += 1
            remove_image = True
        elif len(keep) != len(lines):
            files_changed += 1

        if not apply_changes:
            continue

        if remove_image:
            os.remove(lbl_path)
            for ext in [".jpg", ".jpeg", ".png"]:
                img_path = os.path.join(img_dir, stem + ext)
                if os.path.exists(img_path):
                    os.remove(img_path)
        elif len(keep) != len(lines):
            with open(lbl_path, "w") as f:
                f.write("\n".join(keep) + "\n")

print("tree files checked:", tree_files)
print("tiny boxes removed:", tiny_boxes_removed)
print("images removed because of a huge box:", images_removed_big)
print("images removed because of too many tree boxes:", images_removed_crowd)
print("images removed because nothing was left:", images_removed_empty)
print("label files edited:", files_changed)
if apply_changes:
    print("changes WRITTEN")
else:
    print("dry run only, nothing changed. set apply_changes = True to apply")