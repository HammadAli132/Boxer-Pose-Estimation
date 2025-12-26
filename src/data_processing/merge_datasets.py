#!/usr/bin/env python3
"""
merge_datasets.py (Memory Optimized)

Provides two options for merging datasets:
1. Merge frames from existing videos in /data/raw/frames
2. Merge frames from test annotations (reads full paths from test.json)

Both options:
- Copy images to /data/main_dataset/frames
- Merge annotations into /data/main_dataset/annotations.json
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
    selection = input("\n🔍 Enter directory numbers to merge (space-separated, e.g., '1 2 3'): ").strip()
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
        print(f"\n📄 Processing {dir_name}...")
        
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
            
            # Copy frame to main_dataset
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
                "file_name": fname,  # Store just filename (image is in main_dataset/frames)
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
    print(f"📁 Images saved to: {MAIN_FRAMES}")


def merge_from_test_annotations():
    """Option 2: Merge frames from test annotations (using full paths from test.json)."""
    if not PROCESSED_TEST_ANN.exists():
        print("❌ Test annotations not found at /data/processed/annotations/test.json")
        return
    
    # Load test annotations
    test_ann = load_json_safe(PROCESSED_TEST_ANN)
    if not test_ann:
        print("❌ Failed to load test annotations")
        return
    
    # Define the predefined target categories structure
    TARGET_CATEGORIES = [
        {
            "id": 1,
            "name": "boxer_red",
            "supercategory": "",
            "keypoints": [
                "left_shoulder",
                "right_shoulder",
                "left_hip",
                "right_hip",
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle",
                "left_elbow",
                "left_wrist",
                "right_elbow",
                "right_wrist",
                "neck",
                "nose"
            ],
            "skeleton": [
                [9, 10],
                [1, 3],
                [2, 4],
                [1, 2],
                [3, 4],
                [6, 8],
                [4, 6],
                [5, 7],
                [11, 12],
                [14, 13],
                [1, 13],
                [13, 2],
                [2, 11],
                [1, 9],
                [3, 5]
            ]
        },
        {
            "id": 2,
            "name": "boxer_blue",
            "supercategory": "",
            "keypoints": [
                "left_shoulder",
                "right_shoulder",
                "left_hip",
                "right_hip",
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle",
                "left_elbow",
                "left_wrist",
                "right_elbow",
                "right_wrist",
                "neck",
                "nose"
            ],
            "skeleton": [
                [9, 10],
                [1, 3],
                [2, 4],
                [1, 2],
                [3, 4],
                [6, 8],
                [4, 6],
                [5, 7],
                [11, 12],
                [14, 13],
                [1, 13],
                [13, 2],
                [2, 11],
                [1, 9],
                [3, 5]
            ]
        }
    ]
    
    # Define keypoint mapping
    keypoint_mapping = {
        0: 13,   # nose -> nose (position 13 in target)
        1: 12,   # neck -> neck (position 12 in target)
        2: 0,    # left_shoulder -> left_shoulder (position 0 in target)
        3: 1,    # right_shoulder -> right_shoulder (position 1 in target)
        4: 2,    # left_hip -> left_hip (position 2 in target)
        5: 3,    # right_hip -> right_hip (position 3 in target)
        6: 8,    # left_elbow -> left_elbow (position 8 in target)
        7: 9,    # left_wrist -> left_wrist (position 9 in target)
        8: 10,   # right_elbow -> right_elbow (position 10 in target)
        9: 11,   # right_wrist -> right_wrist (position 11 in target)
        10: 4,   # left_knee -> left_knee (position 4 in target)
        11: 6,   # left_ankle -> left_ankle (position 6 in target)
        12: 5,   # right_knee -> right_knee (position 5 in target)
        13: 7    # right_ankle -> right_ankle (position 7 in target)
    }
    
    print("\n📄 Step 1: Converting keypoint order...")
    
    # Convert all annotations
    category_id_changes_16_to_1 = 0
    category_id_changes_1_to_2 = 0
    
    for annotation in test_ann.get("annotations", []):
        # Step 1: Convert keypoints
        current_keypoints = annotation['keypoints']
        
        # Convert to list of (x, y, v) tuples
        kp_tuples = []
        for i in range(0, len(current_keypoints), 3):
            kp_tuples.append((
                current_keypoints[i],
                current_keypoints[i+1],
                current_keypoints[i+2]
            ))
        
        # Reorder keypoints according to mapping
        reordered_kp_tuples = [None] * 14
        for old_idx, new_idx in keypoint_mapping.items():
            if old_idx < len(kp_tuples):
                reordered_kp_tuples[new_idx] = kp_tuples[old_idx]
        
        # Flatten back to [x, y, v, x, y, v, ...] format
        converted_keypoints = []
        for kp in reordered_kp_tuples:
            if kp is not None:
                converted_keypoints.extend([kp[0], kp[1], kp[2]])
            else:
                converted_keypoints.extend([0, 0, 0])
        
        # Update the annotation with converted keypoints
        annotation['keypoints'] = converted_keypoints
        
        # Step 2: Change category_id from 16 to 1 and 1 to 2
        if annotation['category_id'] == 16:
            annotation['category_id'] = 1
            category_id_changes_16_to_1 += 1
        elif annotation['category_id'] == 1:
            annotation['category_id'] = 2
            category_id_changes_1_to_2 += 1
    
    print(f"✅ Converted keypoints for {len(test_ann.get('annotations', []))} annotations")
    print(f"✅ Changed category_id from 16 to 1 for {category_id_changes_16_to_1} annotations")
    print(f"✅ Changed category_id from 1 to 2 for {category_id_changes_1_to_2} annotations")
    
    # Step 3: Replace categories with predefined structure
    print(f"\n📄 Step 2: Replacing categories with target structure...")
    test_ann['categories'] = TARGET_CATEGORIES
    
    print(f"✅ Replaced categories:")
    for cat in test_ann['categories']:
        print(f"  - {cat['name']} (id={cat['id']})")
    
    # Load or create main annotations
    main_obj = load_or_create_main_annotations()
    max_img_id, max_ann_id, existing_filenames = get_max_ids(main_obj)
    
    # Ensure categories exist (use TARGET_CATEGORIES)
    if not main_obj.get("categories"):
        main_obj["categories"] = TARGET_CATEGORIES
    
    # Create mapping of old image_id to new image_id
    image_id_map = {}
    test_images = test_ann.get("images", [])
    test_annotations = test_ann.get("annotations", [])
    
    images_added = 0
    anns_added = 0
    
    # Copy frames from their original locations (full paths in file_name)
    for img in tqdm(test_images, desc="Copying test frames"):
        # file_name now contains the full path to the original image
        full_path = Path(img["file_name"])
        fname = full_path.name  # Extract just the filename
        
        if not full_path.exists():
            print(f"⚠️ Test frame not found: {full_path}, skipping...")
            continue
        
        # Skip if already exists
        if fname in existing_filenames:
            # Map to existing image ID
            for existing_img in main_obj["images"]:
                if existing_img["file_name"] == fname:
                    image_id_map[img["id"]] = existing_img["id"]
                    break
            continue
        
        # Copy frame to main_dataset
        dst_path = MAIN_FRAMES / fname
        try:
            MAIN_FRAMES.mkdir(parents=True, exist_ok=True)
            shutil.copy2(full_path, dst_path)
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
            "file_name": fname,  # Store just filename (image is in main_dataset/frames)
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
    print(f"📁 Images saved to: {MAIN_FRAMES}")


def main():
    """Main function to handle user choice and execute merge."""
    print("=" * 60)
    print("Dataset Merge Tool (Memory Optimized)")
    print("=" * 60)
    print("\n💡 Note: Images will be copied to /data/main_dataset/frames")
    print("\nChoose merge option:")
    print("1) Merge frames from existing videos (/data/raw/frames)")
    print("2) Merge frames from test annotations (/data/processed/annotations/test.json)")
    
    choice = input("\nEnter 1 or 2: ").strip()
    
    if choice == "1":
        merge_from_video_directories()
    elif choice == "2":
        merge_from_test_annotations()
    else:
        print("❌ Invalid choice. Please enter 1 or 2.")


if __name__ == "__main__":
    main()