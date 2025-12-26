import os
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


# ==================== CONFIGURATION ====================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Input
COCO_ANNOTATION_FILE = PROJECT_ROOT / "data/processed/annotations/test.json"

# Output directory
OUTPUTS_DIR = PROJECT_ROOT / "data/outputs"

# Video settings
FPS = 25
CODEC = 'mp4v'

# Color scheme (BGR format)
COLORS = {
    'boxer_blue': (255, 0, 0),    # Blue
    'boxer_red': (0, 0, 255),     # Red
    'unknown': (128, 128, 128)    # Gray
}

# Styling
BOX_THICKNESS = 2
KEYPOINT_RADIUS = 5
SKELETON_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)

# Custom 14-keypoint skeleton connections (0-indexed)
SKELETON_CONNECTIONS = [
    (8, 9),    # left_elbow to left_wrist
    (0, 2),    # left_shoulder to left_hip
    (1, 3),    # right_shoulder to right_hip
    (0, 1),    # left_shoulder to right_shoulder
    (2, 3),    # left_hip to right_hip
    (5, 7),    # right_knee to right_ankle
    (3, 5),    # right_hip to right_knee
    (4, 6),    # left_knee to left_ankle
    (10, 11),  # right_elbow to right_wrist
    (13, 12),  # nose to neck
    (0, 12),   # left_shoulder to neck
    (12, 1),   # neck to right_shoulder
    (1, 10),   # right_shoulder to right_elbow
    (0, 8),    # left_shoulder to left_elbow
    (2, 4)     # left_hip to left_knee
]


# ==================== HELPER FUNCTIONS ====================
def _find_next_output_number(output_dir: Path, suffix: str = "") -> int:
    """Find the next available output number."""
    pattern = f"output_*{suffix}.mp4"
    existing = list(output_dir.glob(pattern))
    if not existing:
        return 1
    
    numbers = []
    for f in existing:
        try:
            stem = f.stem.replace(suffix, "")
            num_str = stem.split('_')[1]
            numbers.append(int(num_str))
        except (IndexError, ValueError):
            continue
    
    return max(numbers) + 1 if numbers else 1


def _load_coco_annotations(coco_path: Path) -> Dict[str, List[Dict]]:
    """Load COCO format annotations and index by filename (basename)."""
    if not coco_path.exists():
        raise FileNotFoundError(f"COCO annotation file not found: {coco_path}")
    
    with open(coco_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # Create image_id to filename mapping (use basename for matching)
    image_map = {
        img['id']: os.path.basename(img['file_name']) 
        for img in coco_data.get('images', [])
    }
    
    # Group annotations by filename
    annotations_by_file = {}
    for ann in coco_data.get('annotations', []):
        image_id = ann['image_id']
        filename = image_map.get(image_id)
        if filename:
            file_stem = Path(filename).stem
            if file_stem not in annotations_by_file:
                annotations_by_file[file_stem] = []
            annotations_by_file[file_stem].append(ann)
    
    return annotations_by_file


def _get_image_paths_from_coco(coco_path: Path) -> List[Path]:
    """Extract image paths from COCO annotations (handles absolute paths)."""
    with open(coco_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    image_paths = []
    for img in coco_data.get('images', []):
        img_path = Path(img['file_name'])
        
        # Check if path exists (handles both absolute and relative)
        if img_path.exists():
            image_paths.append(img_path)
        else:
            print(f"⚠️ Image not found: {img_path}")
    
    return sorted(image_paths, key=lambda p: p.name)


def _draw_keypoints(frame: np.ndarray, annotation: Dict) -> np.ndarray:
    """Draw keypoints and skeleton on the frame (COCO format)."""
    keypoints = annotation['keypoints']
    bbox = annotation['bbox']
    category_id = annotation['category_id']

    # Determine colors based on category_id
    if category_id == 2:  # boxer_blue
        skeleton_color = (255, 100, 0)    # Bright blue
        bbox_color = (255, 0, 0)          # Blue bbox
        text_bg_color = (200, 0, 0)       # Dark blue
        class_name = "BLUE"
    elif category_id == 1:  # boxer_red
        skeleton_color = (0, 100, 255)    # Bright red
        bbox_color = (0, 0, 255)          # Red bbox
        text_bg_color = (0, 0, 200)       # Dark red
        class_name = "RED"
    else:  # Unknown
        skeleton_color = (128, 128, 128)  # Gray
        bbox_color = (128, 128, 128)
        text_bg_color = (100, 100, 100)
        class_name = "UNKNOWN"
    
    keypoint_color = (0, 255, 0)  # Green keypoints
    text_color = (255, 255, 255)  # White text
    
    # Draw bounding box (x, y, w, h format)
    x, y, w, h = map(int, bbox)
    cv2.rectangle(frame, (x, y), (x + w, y + h), bbox_color, BOX_THICKNESS)
    
    # Parse keypoints (x, y, visibility) triplets
    kpts = []
    for i in range(0, len(keypoints), 3):
        x_kp = keypoints[i]
        y_kp = keypoints[i + 1]
        vis = keypoints[i + 2]
        if vis > 0:  # Only store visible keypoints
            kpts.append((int(x_kp), int(y_kp), vis))
        else:
            kpts.append(None)
    
    # Draw skeleton connections FIRST (so they appear behind keypoints)
    for conn in SKELETON_CONNECTIONS:
        idx1, idx2 = conn
        if idx1 < len(kpts) and idx2 < len(kpts):
            if kpts[idx1] is not None and kpts[idx2] is not None:
                pt1 = (kpts[idx1][0], kpts[idx1][1])
                pt2 = (kpts[idx2][0], kpts[idx2][1])
                cv2.line(frame, pt1, pt2, skeleton_color, SKELETON_THICKNESS, cv2.LINE_AA)
    
    # Draw keypoints on top
    for kpt in kpts:
        if kpt is not None:
            cv2.circle(frame, (kpt[0], kpt[1]), KEYPOINT_RADIUS, keypoint_color, -1, cv2.LINE_AA)
            # Optional: draw a thin border around keypoints for better visibility
            cv2.circle(frame, (kpt[0], kpt[1]), KEYPOINT_RADIUS, (0, 0, 0), 1, cv2.LINE_AA)
    
    # Draw label
    label = f"BOXER {class_name} | KPs:{annotation['num_keypoints']}"
    
    (text_width, text_height), baseline = cv2.getTextSize(
        label, FONT, FONT_SCALE, FONT_THICKNESS
    )
    
    # Draw text background
    cv2.rectangle(
        frame,
        (x, y - text_height - baseline - 8),
        (x + text_width + 4, y - 2),
        text_bg_color,
        -1
    )
    
    # Draw text
    cv2.putText(
        frame,
        label,
        (x + 2, y - baseline - 5),
        FONT,
        FONT_SCALE,
        text_color,
        FONT_THICKNESS,
        cv2.LINE_AA
    )
    
    return frame


# ==================== MAIN VISUALIZATION ====================
def main():
    print("=" * 60)
    print("YOLO KEYPOINT VISUALIZATION")
    print("=" * 60)
    
    # Ask user for annotation file
    print(f"\nDefault annotation file: {COCO_ANNOTATION_FILE.name}")
    user_input = input("Enter annotation filename (or press Enter for default): ").strip()
    
    if user_input:
        annotation_path = PROJECT_ROOT / "data" / "processed" / "annotations" / user_input
        if not annotation_path.exists():
            # Try without adding path
            annotation_path = Path(user_input)
            if not annotation_path.exists():
                print(f"❌ File not found: {user_input}")
                return
    else:
        annotation_path = COCO_ANNOTATION_FILE
    
    if not annotation_path.exists():
        print(f"❌ Annotation file not found: {annotation_path}")
        return
    
    print(f"📄 Using annotations: {annotation_path}")
    
    # Load annotations
    print("Loading annotations...")
    coco_annotations = _load_coco_annotations(annotation_path)
    
    # Get image paths from annotations
    print("Extracting image paths from annotations...")
    frame_paths = _get_image_paths_from_coco(annotation_path)
    
    if not frame_paths:
        print("❌ No valid image paths found in annotations")
        return
    
    print(f"Found {len(frame_paths)} frames")
    
    # Create output directory
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Determine output video path
    output_suffix = f"_{annotation_path.stem}"
    output_number = _find_next_output_number(OUTPUTS_DIR, output_suffix)
    output_path = OUTPUTS_DIR / f"output_{output_number}{output_suffix}.mp4"
    print(f"Output video: {output_path}")
    
    # Initialize video writer
    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        print(f"❌ Could not read first frame: {frame_paths[0]}")
        return
    
    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*CODEC)
    video_writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        FPS,
        (width, height)
    )
    
    if not video_writer.isOpened():
        print("❌ Failed to initialize video writer")
        return
    
    print(f"\nProcessing frames...")
    print(f"Video resolution: {width}x{height}")
    print(f"FPS: {FPS}")
    
    # Process each frame
    frames_with_annotations = 0
    frames_without_annotations = 0
    red_count = 0
    blue_count = 0
    
    for idx, frame_path in enumerate(frame_paths, 1):
        # Read frame
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"⚠️ Could not read frame: {frame_path}")
            continue
        
        # Find annotations for this frame (match by stem)
        frame_stem = frame_path.stem
        frame_annotations = coco_annotations.get(frame_stem, [])
        
        if frame_annotations:
            frames_with_annotations += 1
            for annotation in frame_annotations:
                frame = _draw_keypoints(frame, annotation)
                # Count by category
                if annotation['category_id'] == 1:
                    red_count += 1
                elif annotation['category_id'] == 2:
                    blue_count += 1
        else:
            frames_without_annotations += 1
        
        # Write frame to video
        video_writer.write(frame)
        
        # Progress indicator
        if idx % 50 == 0 or idx == len(frame_paths):
            print(f"  Processed {idx}/{len(frame_paths)} frames...")
    
    # Cleanup
    video_writer.release()
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"Total frames processed: {len(frame_paths)}")
    print(f"Frames with annotations: {frames_with_annotations}")
    print(f"Frames without annotations: {frames_without_annotations}")
    print(f"\nDetection breakdown:")
    print(f"  Boxer RED: {red_count}")
    print(f"  Boxer BLUE: {blue_count}")
    print(f"  Total: {red_count + blue_count}")
    print(f"\nOutput saved to: {output_path}")
    print(f"Video size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()