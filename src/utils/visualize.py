import os
import json
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np


# ==================== CONFIGURATION ====================
# Base project directory (update if needed)
PROJECT_ROOT = Path("D:/Repositories/Boxer-Pose-Estimation").resolve()

# Input directories
TEST_FRAMES_DIR = PROJECT_ROOT / "data/processed/images/test"
ANNOTATIONS_DIR = PROJECT_ROOT / "data/processed/annotations/rf-detr"

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

# Box and text styling
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)  # White text


# ==================== HELPER FUNCTIONS ====================
def _list_image_paths(directory: Path) -> List[Path]:
    """List all image files in directory, sorted."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [p for p in directory.iterdir() if p.suffix.lower() in exts]
    files.sort()
    return files


def _find_next_output_number(output_dir: Path) -> int:
    """Find the next available output number (output_1.mp4, output_2.mp4, etc.)."""
    existing = list(output_dir.glob("output_*.mp4"))
    if not existing:
        return 1
    
    numbers = []
    for f in existing:
        try:
            # Extract number from "output_N.mp4"
            num_str = f.stem.split('_')[1]
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


def _draw_detection(frame: np.ndarray, detection: Dict) -> np.ndarray:
    """Draw a single detection on the frame."""
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


# ==================== MAIN VISUALIZATION ====================
def main():
    print("=" * 60)
    print("DETECTION VISUALIZATION SCRIPT")
    print("=" * 60)
    
    # Verify input directories exist
    if not TEST_FRAMES_DIR.exists():
        raise FileNotFoundError(f"Test frames directory not found: {TEST_FRAMES_DIR}")
    if not ANNOTATIONS_DIR.exists():
        raise FileNotFoundError(f"Annotations directory not found: {ANNOTATIONS_DIR}")
    
    # Create output directory if needed
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get all test frames
    frame_paths = _list_image_paths(TEST_FRAMES_DIR)
    if not frame_paths:
        raise ValueError(f"No images found in {TEST_FRAMES_DIR}")
    
    print(f"\nFound {len(frame_paths)} frames")
    print(f"Input frames: {TEST_FRAMES_DIR}")
    print(f"Annotations: {ANNOTATIONS_DIR}")
    
    # Determine output video path
    output_number = _find_next_output_number(OUTPUTS_DIR)
    output_path = OUTPUTS_DIR / f"output_{output_number}.mp4"
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
    print(f"Frames with detections: {frames_with_annotations}")
    print(f"Frames without detections: {frames_without_annotations}")
    print(f"\nOutput saved to: {output_path}")
    print(f"Video size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise