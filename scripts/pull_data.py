import fiftyone as fo
import fiftyone.zoo as foz

# ---------- 1. COCO classes relevant to blind navigation ----------
# these decide WHICH images get downloaded
coco_classes = [
    "traffic light",
    "stop sign",
    "fire hydrant",
    "bench",
    "parking meter",
    "potted plant",
    "dog",
    "bicycle",
]

# these are the classes we KEEP labels for when exporting.
# the downloaded images also contain people, cars etc, so we keep
# those boxes too, otherwise they would look like empty background
coco_keep = coco_classes + [
    "person",
    "car",
    "bus",
    "truck",
    "motorcycle",
    "chair",
    "dining table",
]

coco = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    label_types=["detections"],
    classes=coco_classes,
    max_samples=4000,
)

coco.export(
    export_dir="Dataset/coco_subset",
    dataset_type=fo.types.YOLOv5Dataset,
    classes=coco_keep,
    overwrite=True,
)
print("coco done, exported to Dataset/coco_subset")

# ---------- 2. Trees from Open Images (COCO has no tree class) ----------
tree_classes = ["Tree"]

tree_keep = tree_classes + [
    "Person",
    "Car",
    "Bus",
    "Truck",
    "Motorcycle",
    "Bicycle",
    "Chair",
    "Table",
]

trees = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=tree_classes,
    max_samples=1500,
)

trees.export(
    export_dir="Dataset/trees_subset",
    dataset_type=fo.types.YOLOv5Dataset,
    classes=tree_keep,
    overwrite=True,
)
print("trees done, exported to Dataset/trees_subset")