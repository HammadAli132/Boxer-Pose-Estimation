#!/usr/bin/env python3
"""
30FPS Frame Extraction + YOLO11x-Pose Tracking Script

Processes a single dataset directory:
  1. Converts annotation frame ranges (30fps space) to native video frame
     indices, then extracts those frames resampled to 30fps.
  2. Runs YOLO11x-pose in track mode (model.track(), persist=True).
  3. Saves annotations to yolo_annotations_tracked.json using NATIVE frame
     numbers as keys — exactly what detect_posters_unfiltered.py expects.

FPS FIX:
    BoxingVI annotations are in 30fps space but V1 videos are 24fps.
    Step 1 converts annotation ranges to native ranges:

        native_frame = round((ann_frame / ANNOTATION_FPS) * native_fps)

    The resampling step then sub-samples native frames back down to 30fps
    (no-op for 24fps since step=max(1.0, 24/30)=1.0, every frame kept).

Usage:
    python run_yolo_30_fps.py \\
        --dir data/DATASET/SixClassBoxingVIDataset/V1 \\
        [--models-dir models] \\
        [--overwrite]
"""

import json
import cv2
import urllib.request
import numpy as np
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
from tqdm import tqdm


# ============================================================================
# PROJECT ROOT
# ============================================================================

def get_project_root() -> Path:
    current_path = Path(__file__).resolve().parent
    for parent in [current_path] + list(current_path.parents):
        if (parent / "data").exists() or (parent / ".git").exists() or (parent / "requirements.txt").exists():
            return parent
    return Path.cwd()

PROJECT_ROOT   = get_project_root()
TARGET_FPS     = 30
ANNOTATION_FPS = 30   # fps space BoxingVI annotations were labeled in


# ============================================================================
# MODEL SETUP
# ============================================================================

def load_generic_model(models_dir: Path):
    from ultralytics import YOLO

    model_path = models_dir / "yolov11x-pose" / "yolo11x-pose.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"⬇️  Downloading YOLO11x-pose …")
        url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt"
        urllib.request.urlretrieve(url, str(model_path))
        print(f"✅ Saved to {model_path}")

    print(f"⚙️  Loading YOLO11x-pose (pre-trained generic) …")
    return YOLO(str(model_path))


# ============================================================================
# SKELETON CONVERSION  (COCO-17 → custom 14-point)
# ============================================================================

def convert_kpts_coco17_to_custom14(kpts: np.ndarray, confs: np.ndarray) -> List[float]:
    """Convert COCO 17-point skeleton to custom 14-point skeleton."""
    l_sho, r_sho = kpts[5], kpts[6]
    neck_xy   = (l_sho + r_sho) / 2.0
    neck_conf = min(confs[5], confs[6])

    custom_ordered_indices = [0, None, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    flat: List[float] = []
    for idx in custom_ordered_indices:
        if idx is None:
            x, y, c = float(neck_xy[0]), float(neck_xy[1]), float(neck_conf)
        else:
            x, y = float(kpts[idx][0]), float(kpts[idx][1])
            c    = float(confs[idx])
        flat.extend([x, y, c])
    return flat


# ============================================================================
# FPS CONVERSION
# ============================================================================

def annotation_to_native_frame(ann_frame: int, native_fps: float) -> int:
    """
    Convert an annotation-space frame index (ANNOTATION_FPS) to native video
    frame index.

    Example (24fps native video):
        ann_frame=300 -> 300/30 = 10.0s -> round(10.0 * 24) = 240
    """
    return round((ann_frame / ANNOTATION_FPS) * native_fps)


def annotation_ranges_to_native(
    frame_ranges: List[Tuple[int, int]],
    native_fps: float,
) -> List[Tuple[int, int]]:
    """Convert a list of annotation-space (start, end) ranges to native frame ranges."""
    return [
        (annotation_to_native_frame(s, native_fps),
         annotation_to_native_frame(e, native_fps))
        for s, e in frame_ranges
    ]


# ============================================================================
# FRAME EXTRACTION AT 30 FPS  (operates on native frame indices)
# ============================================================================

def build_resampled_frame_indices(
    native_fps: float,
    native_ranges: List[Tuple[int, int]],
    target_fps: int = TARGET_FPS,
) -> List[int]:
    """
    Given ranges in NATIVE frame space, compute which native frame indices to
    extract so the resulting sequence plays at ~target_fps.

    step = native_fps / target_fps
      - native 60fps → step 2.0 → every other frame
      - native 24fps → step 0.8 → capped at 1.0 → every frame kept
      - native 30fps → step 1.0 → every frame kept
    """
    step = max(1.0, native_fps / target_fps)
    selected: set = set()

    for native_start, native_end in native_ranges:
        i = 0
        while True:
            abs_idx = native_start + round(i * step)
            if abs_idx > native_end:
                break
            selected.add(abs_idx)
            i += 1

    return sorted(selected)


def extract_frames_at_30fps(
    video_path: Path,
    annotation_ranges: List[Tuple[int, int]],
    output_dir: Path,
    target_fps: int = TARGET_FPS,
) -> List[Tuple[int, Path]]:
    """
    Convert annotation ranges to native, sub-sample to target_fps, extract
    and save frames as JPEGs.

    Returns list of (native_frame_number, saved_path).
    Frame filenames use native frame numbers to match yolo_annotations lookup.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    native_fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"   📽️  Native FPS : {native_fps:.2f}  |  Total frames : {total_frames}")
    print(f"   📐  Annotation FPS assumed : {ANNOTATION_FPS}")
    print(f"   🎯  Target FPS : {target_fps}")

    # ---- KEY FIX: convert annotation ranges → native ranges ----
    native_ranges = annotation_ranges_to_native(annotation_ranges, native_fps)

    frame_indices = build_resampled_frame_indices(native_fps, native_ranges, target_fps)
    print(f"   📸  Frames to extract : {len(frame_indices)}")

    extracted: List[Tuple[int, Path]] = []
    last_seeked = -1

    for frame_num in tqdm(frame_indices, desc="   Extracting frames @30fps", leave=False):
        if frame_num != last_seeked + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        last_seeked = frame_num

        ret, frame = cap.read()
        if not ret:
            print(f"   ⚠️  Could not read frame {frame_num}, skipping.")
            continue

        # Filename uses native frame number so annotation lookup stays consistent
        out_path = output_dir / f"frame_{frame_num:06d}.jpg"
        cv2.imwrite(str(out_path), frame)
        extracted.append((frame_num, out_path))

    cap.release()
    return extracted


# ============================================================================
# YOLO TRACKING
# ============================================================================

def run_yolo_tracking(
    model,
    frame_paths: List[Tuple[int, Path]],
) -> Dict[int, List[dict]]:
    """
    Run YOLO11x-pose in track mode with persist=True.
    frame_number stored in each annotation is the NATIVE frame index.
    Returns:  track_id → [annotation_dict, ...]
    """
    track_histories: Dict[int, List[dict]] = defaultdict(list)
    print(f"\n🔍 Running YOLO tracking on {len(frame_paths)} frames …")

    for frame_num, frame_path in tqdm(frame_paths, desc="   Tracking", leave=False):
        results = model.track(str(frame_path), persist=True, verbose=False)
        result  = results[0]

        if result.boxes is None or result.keypoints is None:
            continue
        if not (hasattr(result.boxes, "id") and result.boxes.id is not None):
            continue

        track_ids  = result.boxes.id.cpu().numpy().astype(int)
        xyxy_boxes = result.boxes.xyxy.cpu().numpy()
        kpts_data  = result.keypoints.data.cpu().numpy()

        for idx, track_id in enumerate(track_ids):
            x1, y1, x2, y2 = xyxy_boxes[idx]
            bw, bh = x2 - x1, y2 - y1

            person_kpts = kpts_data[idx]
            custom_kpts = convert_kpts_coco17_to_custom14(person_kpts[:, :2], person_kpts[:, 2])

            track_histories[int(track_id)].append({
                "frame_number":  frame_num,          # native frame index
                "track_id":      int(track_id),
                "keypoints":     custom_kpts,
                "bbox":          [float(x1), float(y1), float(bw), float(bh)],
                "area":          float(bw * bh),
                "num_keypoints": sum(1 for i in range(2, len(custom_kpts), 3) if custom_kpts[i] > 0),
            })

    return track_histories


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract frames at 30fps (with fps correction) and run YOLO tracking."
    )
    parser.add_argument(
        "--dir", type=str,
        default="data/DATASET/SixClassBoxingVIDataset/V1",
        help="Target video directory (absolute or relative to project root)",
    )
    parser.add_argument(
        "--models-dir", type=str, default="models",
        help="Models directory (relative to project root)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-extract frames and re-run tracking even if outputs already exist",
    )
    args = parser.parse_args()

    dir_path   = Path(args.dir)
    models_dir = Path(args.models_dir)
    if not dir_path.is_absolute():
        dir_path = PROJECT_ROOT / dir_path
    if not models_dir.is_absolute():
        models_dir = PROJECT_ROOT / models_dir

    if not dir_path.exists():
        print(f"❌ Directory not found: {dir_path}")
        return

    ann_path = dir_path / "annotations.json"
    if not ann_path.exists():
        print(f"❌ annotations.json not found in: {dir_path}")
        return

    print(f"\n{'='*70}")
    print(f"📁 Project root    : {PROJECT_ROOT}")
    print(f"📂 Target dir      : {dir_path}")
    print(f"📐 Annotation FPS  : {ANNOTATION_FPS}")
    print(f"🎯 Target FPS      : {TARGET_FPS}")
    print(f"{'='*70}\n")

    with open(ann_path, "r") as f:
        annotations = json.load(f)

    # annotation-space ranges (will be converted inside extract_frames_at_30fps)
    annotation_ranges = [
        (a["start_frame"], a["end_frame"])
        for a in annotations.get("annotations", [])
        if "start_frame" in a and "end_frame" in a
    ]
    if not annotation_ranges:
        print("❌ No valid frame ranges in annotations.json")
        return

    video_files = list(dir_path.glob("*.mp4")) + list(dir_path.glob("*.avi"))
    if not video_files:
        print("❌ No video file found")
        return
    video_path = video_files[0]
    print(f"🎥 Video: {video_path.name}")

    frames_dir   = dir_path / "frames"
    yolo_ann_out = dir_path / "yolo_annotations_tracked.json"

    # -------------------------------------------------------------------------
    # 1. Extract frames
    # -------------------------------------------------------------------------
    if frames_dir.exists() and not args.overwrite:
        print(f"⚠️  /frames already exists — reusing (pass --overwrite to re-extract)")
        extracted_frames = sorted(
            frames_dir.glob("frame_*.jpg"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        extracted_frames = [(int(p.stem.split("_")[1]), p) for p in extracted_frames]
    else:
        print(f"📸 Extracting frames at {TARGET_FPS}fps (with fps correction) → {frames_dir}")
        extracted_frames = extract_frames_at_30fps(
            video_path, annotation_ranges, frames_dir, TARGET_FPS
        )
        if not extracted_frames:
            print("❌ Frame extraction produced no frames.")
            return

    print(f"   ✅ {len(extracted_frames)} frames ready\n")

    # -------------------------------------------------------------------------
    # 2. Load model
    # -------------------------------------------------------------------------
    model = load_generic_model(models_dir)

    # -------------------------------------------------------------------------
    # 3. YOLO tracking
    # -------------------------------------------------------------------------
    print(f"🤖 Running YOLO11x-pose in track mode …")
    track_histories = run_yolo_tracking(model, extracted_frames)

    n_tracks = len(track_histories)
    n_anns   = sum(len(v) for v in track_histories.values())
    print(f"\n📊 Tracking complete: {n_tracks} unique track(s), {n_anns} total detections")

    # -------------------------------------------------------------------------
    # 4. Save yolo_annotations_tracked.json
    # -------------------------------------------------------------------------
    yolo_output = {
        "info": {
            "description": "YOLO 14-point skeleton tracked annotations (fps-corrected, 30fps resampled)",
            "version": "1.2",
            "annotation_fps": ANNOTATION_FPS,
            "target_fps": TARGET_FPS,
        },
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [],
    }

    ann_id = 1
    for track_id, history in track_histories.items():
        for ann in history:
            yolo_output["annotations"].append({
                "id":            ann_id,
                "frame_number":  ann["frame_number"],  # native frame index
                "track_id":      track_id,
                "category_id":   1,
                "keypoints":     ann["keypoints"],
                "bbox":          ann["bbox"],
                "area":          ann["area"],
                "num_keypoints": ann["num_keypoints"],
                "iscrowd":       0,
            })
            ann_id += 1

    with open(yolo_ann_out, "w") as f:
        json.dump(yolo_output, f, indent=2)

    print(f"\n✅ Saved {ann_id - 1} tracked annotation(s) → {yolo_ann_out}")
    print(f"✅ Done! Run detect_posters_unfiltered.py and answer 'y' when prompted.")


if __name__ == "__main__":
    main()