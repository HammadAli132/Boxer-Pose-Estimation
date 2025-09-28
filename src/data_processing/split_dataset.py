import os
import json
import shutil
from tqdm import tqdm

RAW_VIDEOS_DIR = "data/raw/videos"
RAW_FRAMES_DIR = "data/raw/frames"
RAW_ANNOTATIONS_DIR = "data/raw/annotations"
PROCESSED_IMAGES_DIR = "data/processed/images"
PROCESSED_ANNOTATIONS_DIR = "data/processed/annotations"

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV")

def list_videos():
    """Return list of video basenames (without extension)."""
    files = [
        f for f in os.listdir(RAW_VIDEOS_DIR)
        if os.path.isfile(os.path.join(RAW_VIDEOS_DIR, f)) and f.lower().endswith(VIDEO_EXTS)
    ]
    return [os.path.splitext(f)[0] for f in files]

def load_annotations(video_name):
    ann_path = os.path.join(RAW_ANNOTATIONS_DIR, video_name, "annotations.json")
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"Annotations not found: {ann_path}")
    with open(ann_path, "r") as f:
        return json.load(f)

def find_frame_path(file_name, hinted_video=None):
    """
    Try to locate the frame file:
      1) data/raw/frames/{hinted_video}/{file_name}
      2) infer video from filename prefix {prefix}_frame_...
      3) fallback: walk data/raw/frames and find the file
    Returns full path if found, else None.
    """
    # 1) hinted_video
    if hinted_video:
        p = os.path.join(RAW_FRAMES_DIR, hinted_video, file_name)
        if os.path.exists(p):
            return p

    # 2) infer prefix (common naming convention)
    if "_frame_" in file_name:
        prefix = file_name.split("_frame_")[0]
        p = os.path.join(RAW_FRAMES_DIR, prefix, file_name)
        if os.path.exists(p):
            return p

    # 3) full search (slower)
    for root, _, files in os.walk(RAW_FRAMES_DIR):
        if file_name in files:
            return os.path.join(root, file_name)

    return None

def clear_dir(path):
    """Delete directory contents (recreate empty dir)."""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def save_split(ann_data, images, annotations, split_name, hinted_video=None):
    out_images_dir = os.path.join(PROCESSED_IMAGES_DIR, split_name)
    clear_dir(out_images_dir)

    successful_images = []
    missing_files = []

    # Copy frames with progress bar
    for img in tqdm(images, desc=f"Copying -> {split_name}", unit="img"):
        src_path = find_frame_path(img["file_name"], hinted_video)
        if src_path:
            dst_path = os.path.join(out_images_dir, img["file_name"])
            try:
                shutil.copy(src_path, dst_path)
                successful_images.append(img)
            except Exception as e:
                tqdm.write(f"⚠️ Failed copying {img['file_name']}: {e}")
                missing_files.append(img["file_name"])
        else:
            tqdm.write(f"⚠️ Frame not found for {img['file_name']}")
            missing_files.append(img["file_name"])

    # ✅ Skip JSON writing for test split
    out_json_path = None
    split_annotations = []
    if split_name != "test":
        success_ids = {img["id"] for img in successful_images}
        split_annotations = [ann for ann in annotations if ann["image_id"] in success_ids]

        split_json = {
            "info": ann_data.get("info", {}),
            "licenses": ann_data.get("licenses", []),
            "categories": ann_data.get("categories", []),
            "images": successful_images,
            "annotations": split_annotations,
        }
        os.makedirs(PROCESSED_ANNOTATIONS_DIR, exist_ok=True)
        out_json_path = os.path.join(PROCESSED_ANNOTATIONS_DIR, f"{split_name}.json")
        with open(out_json_path, "w") as f:
            json.dump(split_json, f, indent=2)

    return {
        "images_count": len(successful_images),
        "annotations_count": len(split_annotations),
        "missing_count": len(missing_files),
        "missing_files": missing_files,
        "json_path": out_json_path,
        "images_dir": out_images_dir,
    }

def create_splits(ann_data, video_hint, num_frames=None, train_ratio=0.8):
    images = ann_data.get("images", [])
    annotations = ann_data.get("annotations", [])

    # Keep order; optionally limit first N frames
    if num_frames:
        images = images[:num_frames]
        ann_ids = {img["id"] for img in images}
        annotations = [ann for ann in annotations if ann["image_id"] in ann_ids]

    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    # Save splits and gather stats
    train_stats = save_split(ann_data, train_images, annotations, "train", hinted_video=video_hint)
    val_stats = save_split(ann_data, val_images, annotations, "val", hinted_video=video_hint)

    return train_stats, val_stats

def create_test_split(ann_data, video_hint, used_images, num_frames_used, test_count):
    """Create test split from current or other video (frames only, no JSON)."""
    images = ann_data.get("images", [])

    # Case 1: all frames were used
    if num_frames_used is None or num_frames_used >= len(images):
        print("\n⚠️ All frames of current video are already in train/val.")
        videos = list_videos()
        other_videos = [v for v in videos if v != video_hint]
        if not other_videos:
            print("❌ No other videos available for test split.")
            return None
        print("\nAvailable other videos for test split:")
        for i, v in enumerate(other_videos, 1):
            print(f"{i}. {v}")
        try:
            idx = int(input("\nSelect a video index for test: ").strip()) - 1
            test_video = other_videos[idx]
        except Exception:
            print("❌ Invalid selection")
            return None

        # Load annotations from selected test video
        ann_data_test = load_annotations(test_video)

        available = len(ann_data_test["images"])
        if test_count > available:
            print(f"⚠️ Only {available} frames available in {test_video}, reducing test_count to {available}")
            test_count = available

        test_images = ann_data_test["images"][:test_count]
        return save_split(ann_data_test, test_images, [], "test", hinted_video=test_video)

    # Case 2: only N frames used → take frames after N
    else:
        remaining = len(images) - num_frames_used
        if test_count > remaining:
            print(f"⚠️ Only {remaining} frames left after train/val, reducing test_count to {remaining}")
            test_count = remaining

        test_images = images[num_frames_used:num_frames_used+test_count]
        return save_split(ann_data, test_images, [], "test", hinted_video=video_hint)

def main():
    print("\nSplit Dataset — choose source:")
    choice = input("1) Video   2) Merged dataset   (enter 1 or 2): ").strip()

    if choice == "1":
        videos = list_videos()
        if not videos:
            print("❌ No videos found in data/raw/videos/")
            return
        print("\nAvailable videos:")
        for i, v in enumerate(videos, 1):
            print(f"{i}. {v}")
        try:
            idx = int(input("\nSelect a video index: ").strip()) - 1
            video_name = videos[idx]
        except Exception:
            print("❌ Invalid selection")
            return

        try:
            ann_data = load_annotations(video_name)
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return

        frame_choice = input("Use (1) all frames or (2) first N frames? Enter 1/2: ").strip()
        if frame_choice == "1":
            num_frames = None
        else:
            try:
                num_frames = int(input("Enter number of frames (N): ").strip())
                if num_frames <= 0:
                    raise ValueError()
            except Exception:
                print("❌ Invalid number")
                return

        try:
            train_ratio = float(input("Enter train ratio (e.g., 0.8): ").strip())
            if not (0.0 < train_ratio < 1.0):
                raise ValueError()
        except Exception:
            print("❌ Invalid ratio (must be between 0 and 1)")
            return

        print(f"\nSplitting video '{video_name}' — first {num_frames or 'ALL'} frames — train_ratio={train_ratio}\n")
        train_stats, val_stats = create_splits(ann_data, video_name, num_frames=num_frames, train_ratio=train_ratio)

        print(f"\nTrain/Val split completed.")

        # Ask for test split
        try:
            make_test = input("\nDo you want to create a test split? (y/n): ").strip().lower()
            if make_test == "y":
                test_count = int(input("Enter number of test frames (M): ").strip())
                images = ann_data.get("images", [])
                test_stats = create_test_split(ann_data, video_name, images, num_frames, test_count)
                if test_stats:
                    print(f"\nTest: images={test_stats['images_count']}, annotations={test_stats['annotations_count']}, missing={test_stats['missing_count']}")
                    print(f"Test JSON: {test_stats['json_path']}")
                    print(f"Test images dir: {test_stats['images_dir']}")
        except Exception as e:
            print(f"❌ Test split failed: {e}")

    elif choice == "2":
        merged_path = os.path.join(RAW_ANNOTATIONS_DIR, "merged.json")
        if not os.path.exists(merged_path):
            print(f"❌ Merged annotations not found at {merged_path}")
            return
        with open(merged_path, "r") as f:
            ann_data = json.load(f)

        try:
            train_ratio = float(input("Enter train ratio (e.g., 0.8): ").strip())
            if not (0.0 < train_ratio < 1.0):
                raise ValueError()
        except Exception:
            print("❌ Invalid ratio (must be between 0 and 1)")
            return

        print(f"\nSplitting merged dataset — train_ratio={train_ratio}\n")
        train_stats, val_stats = create_splits(ann_data, video_hint=None, num_frames=None, train_ratio=train_ratio)

    else:
        print("❌ Invalid choice")
        return

    # Final summary
    print("\n===== Split Summary =====")
    print(f"Train: images={train_stats['images_count']}, annotations={train_stats['annotations_count']}, missing_frames={train_stats['missing_count']}")
    print(f" Val : images={val_stats['images_count']}, annotations={val_stats['annotations_count']}, missing_frames={val_stats['missing_count']}")
    print(f"\nTrain JSON: {train_stats['json_path']}")
    print(f" Val  JSON: {val_stats['json_path']}")
    print(f"Train images dir: {train_stats['images_dir']}")
    print(f" Val images dir: {val_stats['images_dir']}")
    if train_stats['missing_count'] or val_stats['missing_count']:
        print("\n⚠️ Missing frames sample (first few):")
        print("Train missing:", train_stats['missing_files'][:5])
        print("Val missing:  ", val_stats['missing_files'][:5])
    print("\n✅ Done.")

if __name__ == "__main__":
    main()
