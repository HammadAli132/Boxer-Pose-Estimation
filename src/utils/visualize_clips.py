#!/usr/bin/env python3
"""
Clip Visualization Script

Reads clips from /clips and the matching yolo_annotations.json, overlays
14-point skeletons frame-by-frame, and saves annotated clips to /clips_annotated.

FPS FIX:
    Clip filenames store annotation-space (30fps) start/end frame numbers.
    yolo_annotations.json stores NATIVE frame numbers (from run_yolo_30_fps.py).

    To look up the correct annotation for each clip frame:

        ann_frame   = ann_start + local_idx          (annotation space)
        native_frame = round((ann_frame / 30) * clip_native_fps)   (native space)
        annotations  = frame_annotations[native_frame]

    The clip's own fps (read from VideoCapture) is used as native_fps since
    clip_extraction.py writes clips at native video fps.

Usage:
    python visualize_clips.py [--data-dir data/DATASET] [--num-samples 4] [--all]
    python visualize_clips.py --dir data/DATASET/SixClassBoxingVIDataset/V1 [--num-samples 4]
"""

import json
import cv2
import numpy as np
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional


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
ANNOTATION_FPS = 30   # fps space BoxingVI annotations were labeled in


# ============================================================================
# SKELETON CONFIG
# ============================================================================

SKELETON_CONNECTIONS = [
    (0, 1),   # nose -> neck
    (1, 2),   # neck -> left shoulder
    (1, 3),   # neck -> right shoulder
    (2, 4),   # left shoulder -> left elbow
    (4, 6),   # left elbow -> left wrist
    (3, 5),   # right shoulder -> right elbow
    (5, 7),   # right elbow -> right wrist
    (2, 8),   # left shoulder -> left hip
    (3, 9),   # right shoulder -> right hip
    (8, 9),   # left hip -> right hip
    (8, 10),  # left hip -> left knee
    (10, 12), # left knee -> left ankle
    (9, 11),  # right hip -> right knee
    (11, 13), # right knee -> right ankle
]

PERSON_COLORS = [
    {"skeleton": (0, 255, 255),  "kpt": (0, 200, 255), "bbox": (0, 180, 255)},
    {"skeleton": (50, 100, 255), "kpt": (50, 160, 255), "bbox": (30, 80, 255)},
]
DEFAULT_COLOR = {"skeleton": (180, 180, 180), "kpt": (200, 200, 200), "bbox": (160, 160, 160)}
TEXT_COLOR = (255, 255, 255)
LABEL_BG   = (0, 0, 0)


# ============================================================================
# FPS CONVERSION
# ============================================================================

def annotation_to_native_frame(ann_frame: int, native_fps: float) -> int:
    """
    Convert annotation-space frame index (ANNOTATION_FPS) to native frame index.

    Example (24fps clip):
        ann_frame=301 -> 301/30=10.033s -> round(10.033*24) = 241
    """
    return round((ann_frame / ANNOTATION_FPS) * native_fps)


# ============================================================================
# ANNOTATIONS LOADING
# ============================================================================

def load_frame_annotations(yolo_ann_path: Path) -> Dict[int, List[dict]]:
    """
    Returns:  native_frame_number -> [annotation, ...]
    Frame numbers in yolo_annotations.json are native (set by run_yolo_30_fps.py).
    """
    with open(yolo_ann_path, "r") as f:
        data = json.load(f)

    frame_map: Dict[int, List[dict]] = {}
    for ann in data.get("annotations", []):
        fn = ann["frame_number"]
        frame_map.setdefault(fn, []).append(ann)
    return frame_map


# ============================================================================
# CLIP FRAME-RANGE PARSING
# ============================================================================

def parse_clip_frame_range(clip_path: Path) -> Optional[Tuple[int, int]]:
    """
    Parses (ann_start, ann_end) from the clip filename.
    These are annotation-space (30fps) frame numbers.
    Format: {video_name}_{action}_{ann_start}_{ann_end}.mp4
    """
    parts = clip_path.stem.rsplit("_", 2)
    if len(parts) < 3:
        return None
    try:
        return int(parts[-2]), int(parts[-1])
    except ValueError:
        return None


# ============================================================================
# DRAWING
# ============================================================================

def draw_poses_on_frame(
    frame: np.ndarray,
    annotations: List[dict],
    conf_threshold: float = 0.2,
) -> np.ndarray:
    img = frame.copy()

    for person_order, ann in enumerate(annotations):
        kp_flat = ann["keypoints"]
        bbox    = ann.get("bbox")
        pid     = ann.get("person_id", ann.get("track_id", person_order))
        colors  = PERSON_COLORS[person_order] if person_order < len(PERSON_COLORS) else DEFAULT_COLOR

        kpts = [
            (kp_flat[i], kp_flat[i + 1], kp_flat[i + 2])
            for i in range(0, len(kp_flat), 3)
        ]

        # Bounding box
        if bbox:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(img, (x, y), (x + w, y + h), colors["bbox"], 2)
            label = f"T{pid}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(img, (x, y - lh - 6), (x + lw + 4, y), LABEL_BG, -1)
            cv2.putText(img, label, (x + 2, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1, cv2.LINE_AA)

        # Skeleton limbs
        for p1_idx, p2_idx in SKELETON_CONNECTIONS:
            if p1_idx >= len(kpts) or p2_idx >= len(kpts):
                continue
            x1, y1, v1 = kpts[p1_idx]
            x2, y2, v2 = kpts[p2_idx]
            if v1 > conf_threshold and v2 > conf_threshold:
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
                         colors["skeleton"], 2, cv2.LINE_AA)

        # Keypoints
        for x, y, v in kpts:
            if v > conf_threshold:
                cv2.circle(img, (int(x), int(y)), 5, colors["kpt"], -1)
                cv2.circle(img, (int(x), int(y)), 5, (0, 0, 0), 1)

    return img


# ============================================================================
# SINGLE CLIP ANNOTATION
# ============================================================================

def annotate_clip(
    clip_path: Path,
    frame_annotations: Dict[int, List[dict]],
    output_path: Path,
    conf_threshold: float = 0.2,
) -> bool:
    """
    Reads clip frame-by-frame, overlays poses, writes to output_path.

    Frame mapping (the fps fix):
        - Clip filename stores annotation-space (30fps) start frame (ann_start)
        - For each local frame index in the clip:
              ann_frame    = ann_start + local_idx
              native_frame = round((ann_frame / ANNOTATION_FPS) * clip_fps)
        - Look up native_frame in frame_annotations (which uses native indices)
    """
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        print(f"   ❌ Cannot open: {clip_path.name}")
        return False

    clip_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0   # native fps (clip written at native fps)
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, clip_fps, (width, height))

    ann_range = parse_clip_frame_range(clip_path)  # annotation-space (30fps) start/end
    if ann_range is None:
        print(f"   ⚠️  Could not parse frame range from filename — using sequential indices")

    local_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if ann_range:
            ann_start = ann_range[0]
            # ---- KEY FIX: annotation frame → native frame ----
            ann_frame    = ann_start + local_idx
            native_frame = annotation_to_native_frame(ann_frame, clip_fps)
        else:
            native_frame = local_idx

        anns = frame_annotations.get(native_frame, [])
        annotated_frame = draw_poses_on_frame(frame, anns, conf_threshold)
        writer.write(annotated_frame)
        local_idx += 1

    cap.release()
    writer.release()
    return True


# ============================================================================
# DISCOVERY
# ============================================================================

def find_clip_directories(data_root: Path) -> List[Path]:
    result = []
    for yolo_file in data_root.rglob("yolo_annotations.json"):
        parent    = yolo_file.parent
        clips_dir = parent / "clips"
        if clips_dir.exists() and list(clips_dir.glob("*.mp4")):
            result.append(parent)
    return sorted(result)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Overlay YOLO poses on video clips (fps-corrected).")
    parser.add_argument("--data-dir",       type=str,   default="data/DATASET",
                        help="Root dataset directory (relative to project root)")
    parser.add_argument("--dir",            type=str,   default="data/DATASET/SixClassBoxingVIDataset/V1",
                        help="Specific video directory (must contain /clips and yolo_annotations.json).")
    parser.add_argument("--num-samples",    type=int,   default=10,
                        help="Number of clips to annotate at random (default: 4)")
    parser.add_argument("--all",            action="store_true",
                        help="Annotate every clip found (overrides --num-samples)")
    parser.add_argument("--conf-threshold", type=float, default=0.2,
                        help="Min keypoint confidence to draw (default: 0.2)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve clip pool
    # ------------------------------------------------------------------
    all_clips: List[Tuple[Path, Path]] = []

    if args.dir:
        target_dir = Path(args.dir)
        if not target_dir.is_absolute():
            target_dir = PROJECT_ROOT / target_dir

        if not target_dir.exists():
            print(f"❌ Directory not found: {target_dir}")
            return

        clips_dir     = target_dir / "clips"
        yolo_ann_path = target_dir / "yolo_annotations.json"

        if not yolo_ann_path.exists():
            print(f"❌ yolo_annotations.json not found in: {target_dir}")
            return
        if not clips_dir.exists() or not list(clips_dir.glob("*.mp4")):
            print(f"❌ No .mp4 clips found in: {clips_dir}")
            return

        all_clips = [(target_dir, clip) for clip in sorted(clips_dir.glob("*.mp4"))]
        print(f"📂 Scoped to: {target_dir}  ({len(all_clips)} clip(s) found)")
    else:
        data_root = PROJECT_ROOT / args.data_dir
        if not data_root.exists():
            print(f"❌ Data directory not found: {data_root}")
            return

        clip_dirs = find_clip_directories(data_root)
        if not clip_dirs:
            print("❌ No directories found with /clips and yolo_annotations.json")
            return

        for d in clip_dirs:
            for clip in sorted((d / "clips").glob("*.mp4")):
                all_clips.append((d, clip))

    if not all_clips:
        print("❌ No .mp4 clips found.")
        return

    # ------------------------------------------------------------------
    # Select
    # ------------------------------------------------------------------
    if args.all or args.num_samples >= len(all_clips):
        selected = all_clips
        print(f"🎬 Annotating all {len(all_clips)} clip(s)")
    else:
        selected = random.sample(all_clips, args.num_samples)
        print(f"🎲 Randomly selected {args.num_samples} of {len(all_clips)} clip(s)")

    # ------------------------------------------------------------------
    # Annotate
    # ------------------------------------------------------------------
    total_ok = 0
    for dir_path, clip_path in selected:
        yolo_ann_path     = dir_path / "yolo_annotations.json"
        frame_annotations = load_frame_annotations(yolo_ann_path)
        output_path       = dir_path / "clips_annotated" / clip_path.name

        print(f"\n   📹 {clip_path.parent.parent.name}/{clip_path.name}")

        ok = annotate_clip(clip_path, frame_annotations, output_path, args.conf_threshold)
        if ok:
            total_ok += 1
            print(f"   ✅ Saved → {output_path}")

    print(f"\n✅ Done — {total_ok}/{len(selected)} clips annotated successfully.")


if __name__ == "__main__":
    main()