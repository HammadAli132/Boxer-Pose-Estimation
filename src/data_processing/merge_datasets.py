#!/usr/bin/env python3
"""
merge_datasets.py

Provides two options for merging datasets:
1. Merge frames from existing videos in /data/raw/frames
2. Merge frames from test directory in /data/processed/test

For option 1:
- Lists available video directories
- User selects which directories to use
- User specifies how many frames from each directory
- Copies frames and merges annotations

For option 2:
- Copies frames from processed/test
- Merges annotations from processed/annotations/test.json

Both options merge into /data/main_dataset
"""

import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT / "data" / "raw"
RAW_FRAMES = DATA_RAW / "frames"
RAW_ANNOTATIONS = DATA_RAW / "annotations"

DATA_PROCESSED = PROJECT / "data" / "processed"
PROCESSED_TEST = DATA_PROCESSED / "images" / "test"
PROCESSED_TEST_ANN = DATA_PROCESSED / "annotations" / "test.json"

MAIN_DATA = PROJECT / "data" / "main_dataset"
MAIN_FRAMES = MAIN_DATA / "frames"
MAIN_ANN = MAIN_DATA / "annotations.json"


def load_json_safe(path: Path):
    """Load JSON file safely, return None if not exists."""
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):
    """Save JSON file with proper formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_video_directories():
    """Get list of video directories in /data/raw/frames."""
    if not RAW_FRAMES.exists():
        return []
    return sorted([d.name for d in RAW_FRAMES.iterdir() if d.is_dir()])


def count_frames_in_directory(directory: Path):
    """Count number of image files in a directory."""
    if not directory.exists():
        return 0
    return len([f for f in directory.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']])


def load_or_create_main_annotations():
    """Load existing main annotations or create new empty structure."""
    if MAIN_ANN.exists():
        return load_json_safe(MAIN_ANN)
    
    # Create empty COCO structure
    return {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }


def get_max_ids(main_obj):
    """Get current maximum image_id and annotation_id from main dataset."""
    max_img_id = max([img["id"] for img in main_obj.get("images", [])], default=0)
    max_ann_id = max([ann["id"] for ann in main_obj.get("annotations", [])], default=0)
    existing_filenames = {img["file_name"] for img in main_obj.get("images", [])}
    return max_img_id, max_ann_id, existing_filenames


def merge_from_video_directories():
    """Option 1: Merge frames from raw video directories."""
    video_dirs = get_video_directories()
    
    if not video_dirs:
        print("❌ No video directories found in /data/raw/frames")
        return
    
    # Display available directories with frame counts
    print("\n📁 Available video directories:")
    dir_info = {}
    for i, vdir in enumerate(video_dirs, 1):
        frame_path = RAW_FRAMES / vdir
        count = count_frames_in_directory(frame_path)
        dir_info[i] = {"name": vdir, "count": count, "path": frame_path}
        print(f"  {i}. {vdir} ({count} frames)")
    
    # Get user selection
    selection = input("\n📝 Enter directory numbers to merge (space-separated, e.g., '1 2 3'): ").strip()
    try:
        selected_indices = [int(x) for x in selection.split()]
        selected_dirs = [dir_info[i] for i in selected_indices if i in dir_info]
    except (ValueError, KeyError):
        print("❌ Invalid selection")
        return
    
    if not selected_dirs:
        print("❌ No valid directories selected")
        return
    
    # Get frame counts for each selected directory
    frame_selections = {}
    for dir_data in selected_dirs:
        while True:
            try:
                count = int(input(f"\n📊 How many frames from '{dir_data['name']}'? (max {dir_data['count']}): ").strip())
                if 0 < count <= dir_data['count']:
                    frame_selections[dir_data['name']] = count
                    break
                else:
                    print(f"⚠️ Please enter a number between 1 and {dir_data['count']}")
            except ValueError:
                print("⚠️ Please enter a valid number")
    
    # Load or create main annotations
    main_obj = load_or_create_main_annotations()
    max_img_id, max_ann_id, existing_filenames = get_max_ids(main_obj)
    
    # Ensure categories exist (use from first annotation file)
    if not main_obj.get("categories"):
        for dir_name in frame_selections.keys():
            ann_file = RAW_ANNOTATIONS / dir_name / "annotations.json"
            if ann_file.exists():
                source_ann = load_json_safe(ann_file)
                if source_ann and source_ann.get("categories"):
                    main_obj["categories"] = source_ann["categories"]
                    break
    
    total_images_added = 0
    total_anns_added = 0
    
    # Process each selected directory
    for dir_name, num_frames in frame_selections.items():
        print(f"\n🔄 Processing {dir_name}...")
        
        # Load annotations for this directory
        ann_file = RAW_ANNOTATIONS / dir_name / "annotations.json"
        if not ann_file.exists():
            print(f"⚠️ Annotations not found for {dir_name}, skipping...")
            continue
        
        source_ann = load_json_safe(ann_file)
        if not source_ann:
            print(f"⚠️ Failed to load annotations for {dir_name}, skipping...")
            continue
        
        # Get first N images and their annotations
        source_images = source_ann.get("images", [])[:num_frames]
        source_annotations = source_ann.get("annotations", [])
        
        # Create mapping of old image_id to new image_id
        image_id_map = {}
        frames_dir = RAW_FRAMES / dir_name
        
        # Copy frames and add image entries
        for img in tqdm(source_images, desc=f"Copying frames from {dir_name}"):
            fname = img["file_name"]
            src_path = frames_dir / fname
            
            if not src_path.exists():
                print(f"⚠️ Frame not found: {fname}, skipping...")
                continue
            
            # Skip if already exists
            if fname in existing_filenames:
                print(f"ℹ️ Frame already exists: {fname}, skipping...")
                continue
            
            # Copy frame
            dst_path = MAIN_FRAMES / fname
            try:
                MAIN_FRAMES.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)
            except Exception as e:
                print(f"⚠️ Failed to copy {fname}: {e}")
                continue
            
            # Add image entry with new ID
            old_img_id = img["id"]
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
            existing_filenames.add(fname)
            image_id_map[old_img_id] = max_img_id
            total_images_added += 1
        
        # Add annotations for copied images
        for ann in tqdm(source_annotations, desc=f"Merging annotations from {dir_name}"):
            old_img_id = ann["image_id"]
            if old_img_id not in image_id_map:
                continue  # Skip annotations for images we didn't copy
            
            max_ann_id += 1
            new_ann = ann.copy()
            new_ann["id"] = max_ann_id
            new_ann["image_id"] = image_id_map[old_img_id]
            main_obj["annotations"].append(new_ann)
            total_anns_added += 1
    
    # Save merged annotations
    save_json(MAIN_ANN, main_obj)
    print(f"\n✅ Done! Added {total_images_added} images and {total_anns_added} annotations")
    print(f"📄 Annotations saved to: {MAIN_ANN}")


def merge_from_test_directory():
    """Option 2: Merge frames from processed test directory."""
    if not PROCESSED_TEST.exists():
        print("❌ Test directory not found at /data/processed/images/test")
        return
    
    if not PROCESSED_TEST_ANN.exists():
        print("❌ Test annotations not found at /data/processed/annotations/test.json")
        return
    
    # Load test annotations
    test_ann = load_json_safe(PROCESSED_TEST_ANN)
    if not test_ann:
        print("❌ Failed to load test annotations")
        return
    
    # Load or create main annotations
    main_obj = load_or_create_main_annotations()
    max_img_id, max_ann_id, existing_filenames = get_max_ids(main_obj)
    
    # Ensure categories exist
    if not main_obj.get("categories") and test_ann.get("categories"):
        main_obj["categories"] = test_ann["categories"]
    
    # Create mapping of old image_id to new image_id
    image_id_map = {}
    test_images = test_ann.get("images", [])
    test_annotations = test_ann.get("annotations", [])
    
    images_added = 0
    anns_added = 0
    
    # Copy frames and add image entries
    for img in tqdm(test_images, desc="Copying test frames"):
        fname = img["file_name"]
        src_path = PROCESSED_TEST / fname
        
        if not src_path.exists():
            print(f"⚠️ Test frame not found: {fname}, skipping...")
            continue
        
        # Skip if already exists
        if fname in existing_filenames:
            # Map to existing image ID
            for existing_img in main_obj["images"]:
                if existing_img["file_name"] == fname:
                    image_id_map[img["id"]] = existing_img["id"]
                    break
            continue
        
        # Copy frame
        dst_path = MAIN_FRAMES / fname
        try:
            MAIN_FRAMES.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)
        except Exception as e:
            print(f"⚠️ Failed to copy {fname}: {e}")
            continue
        
        # Add image entry with new ID
        old_img_id = img["id"]
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
        existing_filenames.add(fname)
        image_id_map[old_img_id] = max_img_id
        images_added += 1
    
    # Add annotations for copied images
    for ann in tqdm(test_annotations, desc="Merging test annotations"):
        old_img_id = ann["image_id"]
        if old_img_id not in image_id_map:
            continue  # Skip annotations for images we didn't copy
        
        max_ann_id += 1
        new_ann = ann.copy()
        new_ann["id"] = max_ann_id
        new_ann["image_id"] = image_id_map[old_img_id]
        main_obj["annotations"].append(new_ann)
        anns_added += 1
    
    # Save merged annotations
    save_json(MAIN_ANN, main_obj)
    print(f"\n✅ Done! Added {images_added} images and {anns_added} annotations")
    print(f"📄 Annotations saved to: {MAIN_ANN}")


def main():
    """Main function to handle user choice and execute merge."""
    print("=" * 60)
    print("Dataset Merge Tool")
    print("=" * 60)
    print("\nChoose merge option:")
    print("1) Merge frames from existing videos (/data/raw/frames)")
    print("2) Merge frames from test directory (/data/processed/test)")
    
    choice = input("\nEnter 1 or 2: ").strip()
    
    if choice == "1":
        merge_from_video_directories()
    elif choice == "2":
        merge_from_test_directory()
    else:
        print("❌ Invalid choice. Please enter 1 or 2.")


if __name__ == "__main__":
    main()