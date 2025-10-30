import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np


# ==================== CONFIGURATION ====================
# Base project directory (update if needed)
PROJECT_ROOT = Path("D:/Repositories/Boxer-Pose-Estimation").resolve()

# Input directories
TEST_FRAMES_DIR = PROJECT_ROOT / "data/processed/images/test"
ANNOTATIONS_DIR = PROJECT_ROOT / "data/processed/annotations/rf-detr"
COCO_ANNOTATION_FILE = PROJECT_ROOT / "data/processed/annotations/test.json"

# Output directory
OUTPUTS_DIR = PROJECT_ROOT / "data/outputs"

# Video settings
FPS = 25  # Frames per second for output video
CODEC = 'mp4v'  # Codec for .mp4 output


# Color scheme for bounding boxes (BGR format)
COLORS = {
    'boxer_blue': (255, 0, 0),    # Blue
    'boxer_red': (0, 0, 255),     # Red
    'unknown': (128, 128, 128)    # Gray
}

# Keypoint visualization colors
KEYPOINT_COLOR = (0, 255, 0)  # Green for keypoints
SKELETON_COLOR = (255, 255, 0)  # Cyan for skeleton lines
BBOX_COLOR = (0, 165, 255)  # Orange for bounding boxes

# Box and text styling
BOX_THICKNESS = 2
KEYPOINT_RADIUS = 4
SKELETON_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)  # White text

# Custom 14-keypoint skeleton connections (1-indexed in data, converted to 0-indexed)
# Keypoints: left_shoulder, right_shoulder, left_hip, right_hip, left_knee, right_knee,
#            left_ankle, right_ankle, left_elbow, left_wrist, right_elbow, right_wrist,
#            neck, nose
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
def _list_image_paths(directory: Path) -> List[Path]:
    """List all image files in directory, sorted."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [p for p in directory.iterdir() if p.suffix.lower() in exts]
    files.sort()
    return files


def _find_next_output_number(output_dir: Path, suffix: str = "") -> int:
    """Find the next available output number (output_1.mp4, output_2.mp4, etc.)."""
    pattern = f"output_*{suffix}.mp4"
    existing = list(output_dir.glob(pattern))
    if not existing:
        return 1
    
    numbers = []
    for f in existing:
        try:
            # Extract number from "output_N{suffix}.mp4"
            stem = f.stem.replace(suffix, "")
            num_str = stem.split('_')[1]
            numbers.append(int(num_str))
        except (IndexError, ValueError):
            continue
    
    return max(numbers) + 1 if numbers else 1


def _load_annotation(annotation_path: Path) -> Dict:
    """Load annotation JSON file."""
    if not annotation_path.exists():
        return {'detections': []}
    
    with open(annotation_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _load_coco_annotations(coco_path: Path) -> Dict[str, List[Dict]]:
    """Load COCO format annotations and index by filename."""
    if not coco_path.exists():
        raise FileNotFoundError(f"COCO annotation file not found: {coco_path}")
    
    with open(coco_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # Create image_id to filename mapping
    image_map = {img['id']: img['file_name'] for img in coco_data.get('images', [])}
    
    # Group annotations by filename
    annotations_by_file = {}
    for ann in coco_data.get('annotations', []):
        image_id = ann['image_id']
        filename = image_map.get(image_id)
        if filename:
            # Remove extension to match with frame stem
            file_stem = Path(filename).stem
            if file_stem not in annotations_by_file:
                annotations_by_file[file_stem] = []
            annotations_by_file[file_stem].append(ann)
    
    return annotations_by_file


def _draw_detection(frame: np.ndarray, detection: Dict) -> np.ndarray:
    """Draw a single detection on the frame (RF-DETR format)."""
    bbox = detection['bbox_xyxy']
    x1, y1, x2, y2 = map(int, bbox)
    
    class_name = detection.get('class_name', 'unknown')
    confidence = detection.get('confidence', 0.0)
    track_id = detection.get('track_id', '?')
    
    # Get color for this class
    color = COLORS.get(class_name, COLORS['unknown'])
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
    
    # Prepare label text
    label = f"ID:{track_id} {class_name} {confidence:.2f}"
    
    # Calculate text size for background
    (text_width, text_height), baseline = cv2.getTextSize(
        label, FONT, FONT_SCALE, FONT_THICKNESS
    )
    
    # Draw background rectangle for text
    cv2.rectangle(
        frame,
        (x1, y1 - text_height - baseline - 5),
        (x1 + text_width, y1),
        color,
        -1  # Filled rectangle
    )
    
    # Draw text
    cv2.putText(
        frame,
        label,
        (x1, y1 - baseline - 5),
        FONT,
        FONT_SCALE,
        TEXT_COLOR,
        FONT_THICKNESS,
        cv2.LINE_AA
    )
    
    return frame


def _draw_keypoints(frame: np.ndarray, annotation: Dict) -> np.ndarray:
    """Draw keypoints and skeleton on the frame (COCO format)."""
    keypoints = annotation['keypoints']
    bbox = annotation['bbox']
    
    # Draw bounding box (x, y, w, h format)
    x, y, w, h = map(int, bbox)
    cv2.rectangle(frame, (x, y), (x + w, y + h), BBOX_COLOR, BOX_THICKNESS)
    
    # Parse keypoints (x, y, visibility) triplets
    kpts = []
    for i in range(0, len(keypoints), 3):
        x_kp = keypoints[i]
        y_kp = keypoints[i + 1]
        vis = keypoints[i + 2]
        if vis > 0:  # Only draw visible keypoints
            kpts.append((int(x_kp), int(y_kp), vis))
        else:
            kpts.append(None)
    
    # Draw skeleton connections
    for conn in SKELETON_CONNECTIONS:
        idx1, idx2 = conn
        if idx1 < len(kpts) and idx2 < len(kpts):
            if kpts[idx1] is not None and kpts[idx2] is not None:
                pt1 = (kpts[idx1][0], kpts[idx1][1])
                pt2 = (kpts[idx2][0], kpts[idx2][1])
                cv2.line(frame, pt1, pt2, SKELETON_COLOR, SKELETON_THICKNESS)
    
    # Draw keypoints
    for kpt in kpts:
        if kpt is not None:
            cv2.circle(frame, (kpt[0], kpt[1]), KEYPOINT_RADIUS, KEYPOINT_COLOR, -1)
    
    # Draw annotation ID
    label = f"ID:{annotation['id']} Keypoints:{annotation['num_keypoints']}"
    (text_width, text_height), baseline = cv2.getTextSize(
        label, FONT, FONT_SCALE, FONT_THICKNESS
    )
    
    cv2.rectangle(
        frame,
        (x, y - text_height - baseline - 5),
        (x + text_width, y),
        BBOX_COLOR,
        -1
    )
    
    cv2.putText(
        frame,
        label,
        (x, y - baseline - 5),
        FONT,
        FONT_SCALE,
        TEXT_COLOR,
        FONT_THICKNESS,
        cv2.LINE_AA
    )
    
    return frame


def _get_user_choice() -> str:
    """Prompt user to select annotation format."""
    print("\nSelect annotation format to visualize:")
    print("1. RF-DETR detections (from rf-detr directory)")
    print("2. COCO keypoints (from test.json)")
    
    while True:
        choice = input("\nEnter your choice (1 or 2): ").strip()
        if choice == '1':
            return 'rf-detr'
        elif choice == '2':
            return 'coco'
        else:
            print("Invalid choice. Please enter 1 or 2.")


# ==================== MAIN VISUALIZATION ====================
def main():
    print("=" * 60)
    print("DETECTION & KEYPOINT VISUALIZATION SCRIPT")
    print("=" * 60)
    
    # Get user choice
    annotation_type = _get_user_choice()
    
    # Verify input directories exist
    if not TEST_FRAMES_DIR.exists():
        raise FileNotFoundError(f"Test frames directory not found: {TEST_FRAMES_DIR}")
    
    # Create output directory if needed
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get all test frames
    frame_paths = _list_image_paths(TEST_FRAMES_DIR)
    if not frame_paths:
        raise ValueError(f"No images found in {TEST_FRAMES_DIR}")
    
    print(f"\nFound {len(frame_paths)} frames")
    print(f"Input frames: {TEST_FRAMES_DIR}")
    
    # Load annotations based on type
    coco_annotations = None
    if annotation_type == 'rf-detr':
        if not ANNOTATIONS_DIR.exists():
            raise FileNotFoundError(f"Annotations directory not found: {ANNOTATIONS_DIR}")
        print(f"Annotations: {ANNOTATIONS_DIR}")
        output_suffix = ""
    else:  # coco
        print(f"Annotations: {COCO_ANNOTATION_FILE}")
        coco_annotations = _load_coco_annotations(COCO_ANNOTATION_FILE)
        output_suffix = "_yolo"
    
    # Determine output video path
    output_number = _find_next_output_number(OUTPUTS_DIR, output_suffix)
    output_path = OUTPUTS_DIR / f"output_{output_number}{output_suffix}.mp4"
    print(f"Output video: {output_path}")
    
    # Initialize video writer
    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        raise ValueError(f"Could not read first frame: {frame_paths[0]}")
    
    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*CODEC)
    video_writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        FPS,
        (width, height)
    )
    
    if not video_writer.isOpened():
        raise RuntimeError("Failed to initialize video writer")
    
    print(f"\nProcessing frames...")
    print(f"Video resolution: {width}x{height}")
    print(f"FPS: {FPS}")
    
    # Process each frame
    frames_with_annotations = 0
    frames_without_annotations = 0
    
    for idx, frame_path in enumerate(frame_paths, 1):
        # Read frame
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"Warning: Could not read frame: {frame_path}")
            continue
        
        if annotation_type == 'rf-detr':
            # Find corresponding annotation file
            annotation_path = ANNOTATIONS_DIR / f"{frame_path.stem}.json"
            annotation_data = _load_annotation(annotation_path)
            
            # Draw all detections
            detections = annotation_data.get('detections', [])
            if detections:
                frames_with_annotations += 1
                for detection in detections:
                    frame = _draw_detection(frame, detection)
            else:
                frames_without_annotations += 1
        else:  # coco
            # Find annotations for this frame
            frame_annotations = coco_annotations.get(frame_path.stem, [])
            if frame_annotations:
                frames_with_annotations += 1
                for annotation in frame_annotations:
                    frame = _draw_keypoints(frame, annotation)
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
    print(f"Annotation type: {annotation_type.upper()}")
    print(f"Total frames processed: {len(frame_paths)}")
    print(f"Frames with annotations: {frames_with_annotations}")
    print(f"Frames without annotations: {frames_without_annotations}")
    print(f"\nOutput saved to: {output_path}")
    print(f"Video size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise