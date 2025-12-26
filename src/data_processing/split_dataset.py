import os
import json
from pathlib import Path
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parents[2]
RAW_VIDEOS_DIR = PROJECT / "data" / "raw" / "videos"
RAW_FRAMES_DIR = PROJECT / "data" / "raw" / "frames"
RAW_ANNOTATIONS_DIR = PROJECT / "data" / "raw" / "annotations"
MAIN_DATA = PROJECT / "data" / "main_dataset"
MAIN_FRAMES = MAIN_DATA / "frames"
MAIN_ANN = MAIN_DATA / "annotations.json"
PROCESSED_ANNOTATIONS_DIR = PROJECT / "data" / "processed" / "annotations"

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV")

def list_videos():
    """Return list of video basenames (without extension)."""
    if not RAW_VIDEOS_DIR.exists():
        return []
    files = [
        f for f in os.listdir(RAW_VIDEOS_DIR)
        if os.path.isfile(RAW_VIDEOS_DIR / f) and f.lower().endswith(VIDEO_EXTS)
    ]
    return [os.path.splitext(f)[0] for f in files]

def list_raw_frame_directories():
    """Return list of video directories in data/raw/frames."""
    if not RAW_FRAMES_DIR.exists():
        return []
    return sorted([d.name for d in RAW_FRAMES_DIR.iterdir() if d.is_dir()])

def load_annotations(video_name):
    """Load annotations from raw annotations directory."""
    ann_path = RAW_ANNOTATIONS_DIR / video_name / "annotations.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotations not found: {ann_path}")
    with open(ann_path, "r") as f:
        return json.load(f)

def load_main_annotations():
    """Load annotations from main dataset."""
    if not MAIN_ANN.exists():
        raise FileNotFoundError(f"Main annotations not found: {MAIN_ANN}")
    with open(MAIN_ANN, "r") as f:
        return json.load(f)

def find_frame_path(file_name, hinted_video=None):
    """
    Try to locate the frame file and return its full path:
      1) data/raw/frames/{hinted_video}/{file_name}
      2) infer video from filename prefix {prefix}_frame_...
      3) fallback: walk data/raw/frames and find the file
    Returns full path if found, else None.
    """
    # 1) hinted_video
    if hinted_video:
        p = RAW_FRAMES_DIR / hinted_video / file_name
        if p.exists():
            return str(p)

    # 2) infer prefix (common naming convention)
    if "_frame_" in file_name:
        prefix = file_name.split("_frame_")[0]
        p = RAW_FRAMES_DIR / prefix / file_name
        if p.exists():
            return str(p)

    # 3) full search (slower)
    for root, _, files in os.walk(RAW_FRAMES_DIR):
        if file_name in files:
            return str(Path(root) / file_name)

    return None

def save_split(ann_data, images, annotations, split_name, source_frames_dir=None, hinted_video=None, skip_annotations=False):
    """
    Save a split (train/val/test) WITHOUT copying images.
    Instead, update file_name to include full path to original location.
    - For train/val from merged dataset: source_frames_dir = MAIN_FRAMES
    - For test from raw video: hinted_video = video_name
    - skip_annotations: If True, don't create annotations JSON (for test split)
    """
    successful_images = []
    missing_files = []

    # Update image paths with progress bar
    for img in tqdm(images, desc=f"Processing -> {split_name}", unit="img"):
        fname = img["file_name"]
        
        # Determine source path
        if source_frames_dir:
            # From main dataset or specific directory
            src_path = source_frames_dir / fname
        else:
            # From raw frames (use find_frame_path)
            src_path = find_frame_path(fname, hinted_video)
            if src_path:
                src_path = Path(src_path)
        
        if src_path and src_path.exists():
            # Create new image entry with full path
            img_copy = img.copy()
            img_copy["file_name"] = str(src_path)  # Store full path
            successful_images.append(img_copy)
        else:
            tqdm.write(f"⚠️ Frame not found for {fname}")
            missing_files.append(fname)

    # Save annotations JSON (skip for test split)
    out_json_path = None
    split_annotations = []
    
    if not skip_annotations:
        success_ids = {img["id"] for img in successful_images}
        split_annotations = [ann for ann in annotations if ann["image_id"] in success_ids]

        split_json = {
            "info": ann_data.get("info", {}),
            "licenses": ann_data.get("licenses", []),
            "categories": ann_data.get("categories", []),
            "images": successful_images,
            "annotations": split_annotations,
        }
        
        PROCESSED_ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
        out_json_path = PROCESSED_ANNOTATIONS_DIR / f"{split_name}.json"
        with open(out_json_path, "w") as f:
            json.dump(split_json, f, indent=2)

    return {
        "images_count": len(successful_images),
        "annotations_count": len(split_annotations),
        "missing_count": len(missing_files),
        "missing_files": missing_files,
        "json_path": str(out_json_path) if out_json_path else None,
    }

def create_splits_from_video(ann_data, video_hint, num_frames=None, train_ratio=0.8):
    """Create train/val splits from a single video."""
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

def create_splits_from_merged(ann_data, train_ratio=0.8):
    """Create train/val splits from merged dataset."""
    images = ann_data.get("images", [])
    annotations = ann_data.get("annotations", [])

    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    # Save splits (frames reference from MAIN_FRAMES)
    train_stats = save_split(ann_data, train_images, annotations, "train", source_frames_dir=MAIN_FRAMES)
    val_stats = save_split(ann_data, val_images, annotations, "val", source_frames_dir=MAIN_FRAMES)

    return train_stats, val_stats

def create_test_split_from_video(video_name, start_frame, frame_count):
    """Create test split from a raw video starting at specific frame number."""
    try:
        ann_data = load_annotations(video_name)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return None

    images = ann_data.get("images", [])
    total_available = len(images)
    
    # Validate start frame
    if start_frame < 0 or start_frame >= total_available:
        print(f"❌ Invalid start frame. Must be between 0 and {total_available - 1}")
        return None
    
    # Calculate end frame
    end_frame = start_frame + frame_count
    if end_frame > total_available:
        available_count = total_available - start_frame
        print(f"⚠️ Only {available_count} frames available from frame {start_frame}")
        print(f"   Adjusting to copy frames {start_frame} to {total_available - 1}")
        end_frame = total_available
    
    # Select frames in range [start_frame, end_frame)
    test_images = images[start_frame:end_frame]
    
    print(f"📋 Processing {len(test_images)} frames: frame {start_frame} to frame {end_frame - 1}")
    
    # Skip annotations (will be generated by model at inference time)
    return save_split(ann_data, test_images, [], "test", hinted_video=video_name, skip_annotations=False)

def count_frames_in_directory(directory: Path):
    """Count number of image files in a directory."""
    if not directory.exists():
        return 0
    return len([f for f in directory.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']])

def main():
    print("\n" + "=" * 60)
    print("Dataset Split Tool (Memory Optimized)")
    print("=" * 60)
    print("\n💡 Note: Images will NOT be copied. Annotations will")
    print("   reference original image locations to save disk space.")
    print("\nChoose source for train/val splits:")
    print("1) Single video (from data/raw)")
    print("2) Merged dataset (from data/main_dataset)")
    
    choice = input("\nEnter 1 or 2: ").strip()

    if choice == "1":
        # Option 1: Split from single video
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

        print(f"\n📄 Splitting video '{video_name}' — first {num_frames or 'ALL'} frames — train_ratio={train_ratio}\n")
        train_stats, val_stats = create_splits_from_video(ann_data, video_name, num_frames=num_frames, train_ratio=train_ratio)

    elif choice == "2":
        # Option 2: Split from merged dataset
        try:
            ann_data = load_main_annotations()
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return

        total_images = len(ann_data.get("images", []))
        print(f"\n📊 Main dataset contains {total_images} images")

        try:
            train_ratio = float(input("Enter train ratio (e.g., 0.8): ").strip())
            if not (0.0 < train_ratio < 1.0):
                raise ValueError()
        except Exception:
            print("❌ Invalid ratio (must be between 0 and 1)")
            return

        print(f"\n📄 Splitting merged dataset — train_ratio={train_ratio}\n")
        train_stats, val_stats = create_splits_from_merged(ann_data, train_ratio=train_ratio)

        # Ask for test split from raw video
        make_test = input("\n🔍 Do you want to create a test split from a raw video? (y/n): ").strip().lower()
        if make_test == "y":
            # List available raw frame directories
            frame_dirs = list_raw_frame_directories()
            if not frame_dirs:
                print("❌ No frame directories found in data/raw/frames")
            else:
                print("\n📁 Available video frame directories:")
                dir_info = {}
                for i, vdir in enumerate(frame_dirs, 1):
                    frame_path = RAW_FRAMES_DIR / vdir
                    count = count_frames_in_directory(frame_path)
                    dir_info[i] = {"name": vdir, "count": count}
                    print(f"  {i}. {vdir} ({count} frames)")
                
                try:
                    idx = int(input("\n🔍 Select a video directory for test split: ").strip())
                    if idx not in dir_info:
                        print("❌ Invalid selection")
                    else:
                        selected_video = dir_info[idx]["name"]
                        max_frames = dir_info[idx]["count"]
                        
                        print(f"\n📊 Selected video has {max_frames} frames (indexed 0 to {max_frames - 1})")
                        start_frame = int(input("Enter starting frame number (e.g., 450): ").strip())
                        
                        if start_frame < 0 or start_frame >= max_frames:
                            print(f"❌ Invalid start frame. Must be between 0 and {max_frames - 1}")
                        else:
                            remaining = max_frames - start_frame
                            frame_count = int(input(f"Enter number of frames to copy (max {remaining} from frame {start_frame}): ").strip())
                            
                            if frame_count <= 0:
                                print("❌ Frame count must be positive")
                            else:
                                print(f"\n📄 Creating test split from {selected_video}...")
                                print(f"   Processing frames {start_frame} to {start_frame + frame_count - 1}\n")
                                test_stats = create_test_split_from_video(selected_video, start_frame, frame_count)
                                if test_stats:
                                    print(f"\n✅ Test split created!")
                                    print(f"Test: images={test_stats['images_count']}, missing={test_stats['missing_count']}")
                except ValueError:
                    print("❌ Invalid input. Please enter valid numbers.")
                except Exception as e:
                    print(f"❌ Test split failed: {e}")

    else:
        print("❌ Invalid choice")
        return

    # Final summary
    print("\n" + "=" * 60)
    print("Split Summary")
    print("=" * 60)
    print(f"Train: images={train_stats['images_count']}, annotations={train_stats['annotations_count']}, missing_frames={train_stats['missing_count']}")
    print(f" Val : images={val_stats['images_count']}, annotations={val_stats['annotations_count']}, missing_frames={val_stats['missing_count']}")
    print(f"\nTrain JSON: {train_stats['json_path']}")
    print(f" Val  JSON: {val_stats['json_path']}")
    print(f"\n💡 Images are referenced from their original locations")
    print(f"   (no copying performed to save disk space)")
    
    if train_stats['missing_count'] or val_stats['missing_count']:
        print("\n⚠️ Missing frames sample (first few):")
        if train_stats['missing_count']:
            print("Train missing:", train_stats['missing_files'][:5])
        if val_stats['missing_count']:
            print("Val missing:  ", val_stats['missing_files'][:5])
    
    print("\n✅ Done.")

if __name__ == "__main__":
    main()