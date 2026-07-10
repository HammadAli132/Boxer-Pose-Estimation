#!/usr/bin/env python3
"""
Clip Extraction Script

Traverses data/DATASET/ to find all annotations.json files, then for each
annotation entry extracts the corresponding frame range from the source video
and saves it as a clip under a /clips directory beside the video.

Clip naming convention:
    {video_name}_{action}_{start_frame}_{end_frame}.mp4

  start_frame / end_frame in the filename are kept in ANNOTATION space (30fps)
  so they stay consistent with annotations.json for downstream lookups.

FPS FIX:
    BoxingVI annotations were labeled in 30fps space but V1 videos are 24fps.
    Before seeking, every annotation frame index is converted:

        native_frame = round((ann_frame / ANNOTATION_FPS) * native_fps)

    ann_frame=300 @24fps → 300/30=10.0s → round(10.0*24) = frame 240 ✓

Usage:
    python clip_extraction.py [--data-dir data/DATASET] [--overwrite]
"""

import json
import cv2
import argparse
from pathlib import Path
from typing import List
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
ANNOTATION_FPS = 30   # fps space BoxingVI annotations were labeled in


# ============================================================================
# HELPERS
# ============================================================================

def sanitize_for_filename(text: str) -> str:
    """Replace spaces and special characters with underscores for safe filenames."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(text)).strip("_")


def annotation_to_native_frame(ann_frame: int, native_fps: float) -> int:
    """
    Convert an annotation-space frame index (30fps) to native video frame index.

    Example (24fps native video):
        ann_frame=300 -> 300/30 = 10.0s -> round(10.0 * 24) = 240
    """
    return round((ann_frame / ANNOTATION_FPS) * native_fps)


# ============================================================================
# CORE: CLIP EXTRACTION
# ============================================================================

def extract_clips_from_annotations(
    video_path: Path,
    annotations: dict,
    clips_dir: Path,
    overwrite: bool = False,
) -> List[Path]:
    """
    Creates one video clip per annotation entry.

    start_frame/end_frame in annotations.json are in 30fps annotation space.
    They are converted to native video frame indices before seeking so the
    clip contains exactly the right temporal content.

    Returns a list of successfully created clip paths.
    """
    clips_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"   📽️  Native FPS : {native_fps:.2f}  |  Annotation FPS assumed : {ANNOTATION_FPS}")

    video_name  = sanitize_for_filename(video_path.stem)
    ann_entries = annotations.get("annotations", [])
    created: List[Path] = []

    for ann in tqdm(ann_entries, desc="   Extracting clips", leave=False):
        ann_start = ann.get("start_frame")
        ann_end   = ann.get("end_frame")
        action    = ann.get("class", "unknown")

        if ann_start is None or ann_end is None:
            print(f"   ⚠️  Skipping annotation with missing start/end frame: {ann}")
            continue

        # ---- KEY FIX: convert annotation frames → native video frames ----
        native_start = annotation_to_native_frame(ann_start, native_fps)
        native_end   = annotation_to_native_frame(ann_end,   native_fps)

        # Filename keeps annotation-space indices for traceability
        action_safe = sanitize_for_filename(action)
        clip_name   = f"{video_name}_{action_safe}_{ann_start}_{ann_end}.mp4"
        clip_path   = clips_dir / clip_name

        if not overwrite and clip_path.exists():
            print(f"   ⏭️  Already exists, skipping: {clip_name}")
            created.append(clip_path)
            continue

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(clip_path), fourcc, native_fps, (width, height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, native_start)
        for frame_num in range(native_start, native_end + 1):
            ret, frame = cap.read()
            if not ret:
                print(f"   ⚠️  Could not read native frame {frame_num} — stopping clip early")
                break
            writer.write(frame)

        writer.release()
        created.append(clip_path)
        print(f"   🎬 {clip_name}  "
              f"(ann {ann_start}-{ann_end} → native {native_start}-{native_end}, "
              f"{native_end - native_start + 1} frames)")

    cap.release()
    return created


# ============================================================================
# DIRECTORY PROCESSING
# ============================================================================

def process_directory(ann_path: Path, overwrite: bool) -> bool:
    dir_path  = ann_path.parent
    clips_dir = dir_path / "clips"

    print(f"\n{'='*70}")
    print(f"📂 {dir_path}")

    try:
        with open(ann_path, "r") as f:
            annotations = json.load(f)
    except Exception as e:
        print(f"   ❌ Failed to load annotations.json: {e}")
        return False

    video_files = list(dir_path.glob("*.mp4")) + list(dir_path.glob("*.avi"))
    if not video_files:
        print(f"   ❌ No video file found.")
        return False
    video_path = video_files[0]
    print(f"   🎥 Video: {video_path.name}")

    ann_entries = [
        a for a in annotations.get("annotations", [])
        if "start_frame" in a and "end_frame" in a
    ]
    if not ann_entries:
        print(f"   ❌ No valid annotation entries (need start_frame + end_frame).")
        return False

    clips = extract_clips_from_annotations(video_path, annotations, clips_dir, overwrite)
    print(f"   ✅ {len(clips)} clip(s) saved → {clips_dir}")
    return True


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract video clips from annotated frame ranges with fps correction."
    )
    parser.add_argument(
        "--dir", type=str,
        default="data/DATASET/SixClassBoxingVIDataset/V1",
        help="Target video directory (absolute or relative to project root)",
    )
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-create clips even if they already exist")
    args = parser.parse_args()

    target_dir = Path(args.dir)
    if not target_dir.is_absolute():
        target_dir = PROJECT_ROOT / target_dir

    if not target_dir.exists():
        print(f"❌ Directory not found: {target_dir}")
        return

    ann_path = target_dir / "annotations.json"
    if not ann_path.exists():
        print(f"❌ annotations.json not found in: {target_dir}")
        return

    print(f"📁 Project root    : {PROJECT_ROOT}")
    print(f"📂 Target dir      : {target_dir}")
    print(f"📐 Annotation FPS  : {ANNOTATION_FPS}")

    process_directory(ann_path, args.overwrite)


if __name__ == "__main__":
    main()