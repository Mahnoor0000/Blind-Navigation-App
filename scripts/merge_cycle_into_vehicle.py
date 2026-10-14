# Turns every cycle box (class 2) into vehicle (class 1) in merged_dataset_v2.
#
# Why: the model confused cycle and vehicle (VIAD scooters were labelled cycle,
# motorcycles vehicle). For a walking user both mean "moving thing, step aside".
#
# Class ids do NOT shift. Class 2 stays in data.yaml with zero boxes, so pothole is
# still 20, stairs still 5, and fusion_engine.py keeps working.
#
# Run from the FYP1 folder:
#   python scripts/merge_cycle_into_vehicle.py            (dry run, only counts)
#   set APPLY = True and run again                         (backs up labels, then changes them)

import os
import shutil

DST_DIR = "Dataset/merged_dataset_v2"
FROM_ID = 2   # cycle
TO_ID = 1     # vehicle
APPLY = True

labels_root = os.path.join(DST_DIR, "labels")
backup_root = os.path.join(DST_DIR, "labels_backup_before_cycle_merge")

files_changed = 0
boxes_changed = 0
per_split = {}

if APPLY:
    if os.path.exists(backup_root):
        print("backup already exists, not overwriting it:", backup_root)
    else:
        print("backing up labels to", backup_root, "...")
        shutil.copytree(labels_root, backup_root)

for split in ["train", "valid", "test"]:
    folder = os.path.join(labels_root, split)
    if not os.path.isdir(folder):
        continue
    n = 0
    for fname in os.listdir(folder):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(folder, fname)
        with open(path) as f:
            lines = f.read().splitlines()
        new_lines = []
        changed = 0
        for line in lines:
            parts = line.split()
            if parts and parts[0] == str(FROM_ID):
                parts[0] = str(TO_ID)
                changed += 1
            new_lines.append(" ".join(parts))
        if changed:
            files_changed += 1
            boxes_changed += changed
            n += changed
            if APPLY:
                with open(path, "w") as f:
                    f.write("\n".join(new_lines) + "\n")
    per_split[split] = n

print("cycle boxes per split:", per_split)
print("files with cycle boxes:", files_changed)
print("boxes", "changed" if APPLY else "that would change", ":", boxes_changed)
if APPLY:
    print("done. Delete Dataset/merged_subset and rebuild it before training.")
    print("to undo: delete labels/ and rename the backup folder back to labels")
else:
    print("DRY RUN - nothing changed. Set APPLY = True to apply.")