#!/usr/bin/env python3
"""
update_main_dataset.py

- Prompts user to append either (train+val+test) OR only test splits into main dataset.
- Copies images from data/processed/images/{split}/ to data/main_dataset/frames/ (keeps filenames).
- Merges COCO annotations into data/main_dataset/main_annotations.json (creates if missing).
- Ensures unique image_id and annotation_id; avoids duplicate file_name copies.
- Prints a summary of how many images/annotations were added.
"""

import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = PROJECT / "data" / "processed"
PROCESSED_IMAGES = DATA_PROCESSED / "images"
PROCESSED_ANNOTS = DATA_PROCESSED / "annotations"

MAIN_DATA = PROJECT / "data" / "main_dataset"
MAIN_FRAMES = MAIN_DATA / "frames"
MAIN_ANN = MAIN_DATA / "main_annotations.json"

SPLITS = {
    "train": PROCESSED_IMAGES / "train",
    "val": PROCESSED_IMAGES / "val",
    "test": PROCESSED_IMAGES / "test"
}


def prompt_choice():
    print("Choose append mode:")
    print("1) Append images & annotations from train + val + test")
    print("2) Append images & annotations from ONLY test")
    c = input("Enter 1 or 2: ").strip()
    return c == "1"


def load_json_safe(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_main_annotations_base(example_annot_path: Path):
    """
    If main_annotations.json doesn't exist, create a base skeleton using example_annot_path (train.json or val.json or test.json).
    """
    if MAIN_ANN.exists():
        return

    template = load_json_safe(example_annot_path)
    if template is None:
        # create minimal skeleton
        base = {
            "info": {},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }
    else:
        base = {
            "info": template.get("info", {}),
            "licenses": template.get("licenses", []),
            "images": [],
            "annotations": [],
            "categories": template.get("categories", [])
        }

    MAIN_DATA.mkdir(parents=True, exist_ok=True)
    MAIN_FRAMES.mkdir(parents=True, exist_ok=True)
    with open(MAIN_ANN, "w", encoding="utf-8") as f:
        json.dump(base, f, indent=2)


def map_existing_main_ids(main_obj):
    img_name_to_id = {img["file_name"]: img["id"] for img in main_obj.get("images", [])}
    max_img_id = max([img["id"] for img in main_obj.get("images", [])], default=0)
    max_ann_id = max([ann["id"] for ann in main_obj.get("annotations", [])], default=0)
    return img_name_to_id, max_img_id, max_ann_id


def copy_and_merge(splits_to_merge):
    # ensure main exists
    # use an example annotations JSON to seed main file if not present
    example_json = PROCESSED_ANNOTS / "train.json"
    if not example_json.exists():
        # fallback
        for cand in ["val.json", "test.json"]:
            p = PROCESSED_ANNOTS / cand
            if p.exists():
                example_json = p
                break

    ensure_main_annotations_base(example_json)

    main_obj = load_json_safe(MAIN_ANN)
    if main_obj is None:
        main_obj = {
            "info": {},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }
    img_name_to_id, max_img_id, max_ann_id = map_existing_main_ids(main_obj)

    added_images = 0
    added_anns = 0

    for split in splits_to_merge:
        ann_file = PROCESSED_ANNOTS / f"{split}.json"
        if not ann_file.exists():
            print(f"⚠️ {ann_file} not found — skipping {split}.")
            continue

        split_json = load_json_safe(ann_file)
        if split_json is None:
            continue

        images = split_json.get("images", [])
        annotations = split_json.get("annotations", [])
        imgid_map = {}  # old_image_id -> new_image_id

        # copy images and create image entries
        for img in tqdm(images, desc=f"Copying images ({split})"):
            fname = img["file_name"]
            src_path = SPLITS[split] / fname
            if not src_path.exists():
                # try to find image anywhere under processed images (fallback)
                found = None
                for root, _, files in os.walk(PROCESSED_IMAGES):
                    if fname in files:
                        found = Path(root) / fname
                        break
                if found:
                    src_path = found
                else:
                    print(f"⚠️ Image not found for {fname}, skipping image.")
                    continue

            # if filename already in main dataset -> map to existing id and skip physical copy
            if fname in img_name_to_id:
                new_id = img_name_to_id[fname]
                imgid_map[img["id"]] = new_id
                continue

            # copy file to main frames folder
            dst_path = MAIN_FRAMES / fname
            try:
                shutil.copy2(src_path, dst_path)
            except Exception as e:
                print(f"⚠️ Failed to copy {src_path} -> {dst_path}: {e}")
                continue

            # assign new image id
            max_img_id += 1
            new_img_entry = {
                "id": max_img_id,
                "width": img.get("width", 0),
                "height": img.get("height", 0),
                "file_name": fname,
                "license": img.get("license", 0),
                "flickr_url": img.get("flickr_url", ""),
                "coco_url": img.get("coco_url", ""),
                "date_captured": img.get("date_captured", 0),
            }
            main_obj["images"].append(new_img_entry)
            img_name_to_id[fname] = max_img_id
            imgid_map[img["id"]] = max_img_id
            added_images += 1

        # now remap and append annotations for this split
        for ann in tqdm(annotations, desc=f"Merging annotations ({split})"):
            old_img_id = ann["image_id"]
            if old_img_id not in imgid_map:
                # skip annotations for images we didn't add
                continue
            max_ann_id += 1
            new_ann = ann.copy()
            new_ann["id"] = max_ann_id
            new_ann["image_id"] = imgid_map[old_img_id]
            main_obj["annotations"].append(new_ann)
            added_anns += 1

    # Save merged main annotations
    with open(MAIN_ANN, "w", encoding="utf-8") as f:
        json.dump(main_obj, f, indent=2)

    return added_images, added_anns


def main():
    append_all = prompt_choice()
    splits = ["test"] if not append_all else ["train", "val", "test"]
    print("Selected splits:", splits)
    added_images, added_anns = copy_and_merge(splits)
    print(f"\n✅ Done. Added {added_images} images and {added_anns} annotations to {MAIN_ANN}")


if __name__ == "__main__":
    main()
