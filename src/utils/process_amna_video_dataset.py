#!/usr/bin/env python3
"""
YOLOv11-Pose Video Dataset Processing Script

This script:
1. Traverses the data/DATASET/ directory to find all annotations.json files
2. Extracts frames from videos based on annotation frame ranges
3. Routes Olympic datasets to custom YOLOv11x-pose and others to YOLO11x-pose
4. Runs inference and saves 14-point skeleton annotations with CONFIDENCE SCORES
5. Applies NMS to custom model outputs to ensure max 2 boxers (no duplicates)
"""

import sys
import os
import json
import cv2
import torch
import urllib.request
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import numpy as np

# ============================================================================
# PROJECT ROOT DETECTION
# ============================================================================

def get_project_root() -> Path:
    current_path = Path(__file__).resolve().parent
    for parent in [current_path] + list(current_path.parents):
        if (parent / "data").exists() or (parent / ".git").exists() or (parent / "requirements.txt").exists():
            return parent
    return Path.cwd()

PROJECT_ROOT = get_project_root()
print(f"📁 Project Root: {PROJECT_ROOT}")

# ============================================================================
# YOLO MODEL SETUP
# ============================================================================

def setup_yolo_models(models_dir: Path):
    """
    Load custom YOLOv11m-pose and download/load pre-trained YOLO11x-pose.
    """
    from ultralytics import YOLO
    
    # 1. Load Custom Olympic Model
    custom_model_path = models_dir / "yolov11x-pose" / "best.pt"
    if not custom_model_path.exists():
        raise FileNotFoundError(f"❌ Custom Olympic model missing at: {custom_model_path}")
    
    print(f"⚙️  Loading Custom Olympic Model (YOLOv11x-pose)...")
    model_olympic = YOLO(str(custom_model_path))
    
    # 2. Setup/Download Pre-trained Generic Model
    generic_dir = models_dir / "yolov11x-pose"
    generic_dir.mkdir(parents=True, exist_ok=True)
    generic_model_path = generic_dir / "yolo11x-pose.pt"
    
    if not generic_model_path.exists():
        print(f"⬇️  Downloading pre-trained YOLO11x-pose (might take a minute)...")
        url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt"
        urllib.request.urlretrieve(url, str(generic_model_path))
        print(f"✅ Downloaded to {generic_model_path}")
        
    print(f"⚙️  Loading Pre-trained Generic Model (YOLO11x-pose)...")
    model_generic = YOLO(str(generic_model_path))
    
    return model_olympic, model_generic

# ============================================================================
# SKELETON CONVERSION
# ============================================================================

def convert_kpts_coco17_to_custom14(kpts: np.ndarray, confs: np.ndarray) -> List[float]:
    """Convert COCO 17-point skeleton to custom 14-point skeleton"""
    # Calculate Neck (midpoint of shoulders)
    l_sho, r_sho = kpts[5], kpts[6]
    neck_xy = (l_sho + r_sho) / 2.0
    neck_conf = min(confs[5], confs[6])

    # Custom 14-point order (same as before)
    custom_ordered_indices = [0, None, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    
    flat_list = []
    for idx in custom_ordered_indices:
        if idx is None:
            x, y = neck_xy
            c = neck_conf
        else:
            x, y = kpts[idx]
            c = confs[idx]
            
        # Store x, y, confidence (float)
        # FIX: Explicitly cast 'c' to python float to avoid JSON serialization errors
        flat_list.extend([float(x), float(y), float(c)])
        
    return flat_list

def format_native_custom14(kpts: np.ndarray, confs: np.ndarray) -> List[float]:
    """Format the 14-point skeleton natively output by the custom model"""
    flat_list = []
    for i in range(len(kpts)):  # Should be 14
        x, y = kpts[i]
        c = confs[i]
        # FIX: Explicitly cast 'c' to python float
        flat_list.extend([float(x), float(y), float(c)])
    return flat_list

# ============================================================================
# HELPER: NMS / FILTERING
# ============================================================================

def calculate_iou(box1, box2):
    """Calculates Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    if union_area == 0: return 0
    return inter_area / union_area

def filter_custom_model_detections(boxes, confs, iou_thresh=0.5, max_dets=2):
    """
    Filters detections for the custom model to remove duplicates.
    1. Sorts by confidence.
    2. Removes boxes that overlap significantly (IoU > thresh) with higher confidence boxes.
    3. Keeps at most 'max_dets' (2) boxes.
    """
    if len(boxes) == 0: return []
    
    # Sort indices by confidence (descending)
    sorted_indices = np.argsort(confs)[::-1]
    
    keep_indices = []
    
    for i in sorted_indices:
        current_box = boxes[i]
        
        # Check overlap with already kept boxes
        is_duplicate = False
        for kept_idx in keep_indices:
            kept_box = boxes[kept_idx]
            if calculate_iou(current_box, kept_box) > iou_thresh:
                is_duplicate = True
                break
        
        if not is_duplicate:
            keep_indices.append(i)
            
        # Stop if we found enough people
        if len(keep_indices) >= max_dets:
            break
            
    return keep_indices

# ============================================================================
# VIDEO PROCESSING
# ============================================================================

def extract_frames_from_video(video_path: Path, frame_ranges: List[Tuple[int, int]], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frames_to_extract = set()
    for start, end in frame_ranges:
        frames_to_extract.update(range(start, end + 1))
    frames_to_extract = sorted(frames_to_extract)
    
    extracted_frames = []
    current_frame = 0
    
    for frame_num in tqdm(frames_to_extract, desc="   Extracting frames", leave=False):
        if frame_num != current_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            current_frame = frame_num
        
        ret, frame = cap.read()
        if not ret: continue
        
        frame_path = output_dir / f"frame_{frame_num:06d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        extracted_frames.append((frame_num, frame_path))
        current_frame += 1
    
    cap.release()
    return extracted_frames

# ============================================================================
# YOLO INFERENCE
# ============================================================================

def run_yolo_inference(model, frame_paths: List[Tuple[int, Path]], is_custom_model: bool) -> Dict[int, List]:
    """Run Ultralytics YOLO inference on frames and extract multiple skeletons."""
    annotations = {}
    
    for frame_num, frame_path in tqdm(frame_paths, desc="   Running YOLO", leave=False):
        results = model(str(frame_path), verbose=False)
        result = results[0]
        
        frame_annotations = []
        
        if result.boxes is not None and result.keypoints is not None and len(result.boxes) > 0:
            xyxy_boxes = result.boxes.xyxy.cpu().numpy()
            kpts_data = result.keypoints.data.cpu().numpy()
            box_confs = result.boxes.conf.cpu().numpy() # Get box confidence for sorting

            indices = []
            
            # --- FILTERING LOGIC ---
            if is_custom_model:
                # Custom Model: Use NMS to remove duplicates and keep max 2 boxers
                indices = filter_custom_model_detections(xyxy_boxes, box_confs, iou_thresh=0.5, max_dets=2)
            else:
                # Generic Model: Old logic to filter posters (Keep second largest area if > 1 detection)
                # Note: This was your previous logic for dealing with the poster in the generic dataset
                if len(xyxy_boxes) > 1:
                    areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in xyxy_boxes]
                    sorted_areas = sorted(areas, reverse=True)
                    if len(sorted_areas) > 1:
                        second_largest_area = sorted_areas[1]
                        indices = [i for i, area in enumerate(areas) if area == second_largest_area]
                    else:
                        indices = [int(np.argmax(areas))]
                else:
                    indices = range(len(xyxy_boxes))

            # --- PROCESS SELECTED DETECTIONS ---
            for person_idx in indices:
                # 1. Bounding Box
                x1, y1, x2, y2 = xyxy_boxes[person_idx]
                bbox_w, bbox_h = x2 - x1, y2 - y1
                bbox = [float(x1), float(y1), float(bbox_w), float(bbox_h)]
                area = float(bbox_w * bbox_h)
                
                # 2. Keypoints
                person_kpts = kpts_data[person_idx]
                coords = person_kpts[:, :2]
                confs = person_kpts[:, 2]
                
                # 3. Choose the right formatter based on the model!
                if is_custom_model:
                    # Model already outputs 14 points
                    custom_kpts = format_native_custom14(coords, confs)
                else:
                    # Model outputs 17 points, needs conversion to 14
                    custom_kpts = convert_kpts_coco17_to_custom14(coords, confs)
                
                annotation = {
                    "keypoints": custom_kpts,
                    "bbox": bbox,
                    "area": area,
                    "num_keypoints": sum(1 for i in range(2, len(custom_kpts), 3) if custom_kpts[i] > 0),
                    "person_id": int(person_idx) # Ensure ID is int
                }
                frame_annotations.append(annotation)
                
        annotations[frame_num] = frame_annotations
        
    return annotations

# ============================================================================
# MAIN LOGIC
# ============================================================================

def process_single_directory(ann_path: Path, model_olympic, model_generic, overwrite: bool = True):
    dir_path = ann_path.parent
    print(f"\n{'='*80}\nProcessing: {dir_path}")
    
    frames_dir = dir_path / "frames"
    yolo_ann_path = dir_path / "yolo_annotations.json"
    
    if not overwrite and yolo_ann_path.exists():
        print(f"⏭️  Skipping (yolo_annotations.json already exists).")
        return True
    
    try:
        with open(ann_path, 'r') as f: annotations = json.load(f)
    except Exception as e: return False
    
    video_files = list(dir_path.glob("*.mp4")) + list(dir_path.glob("*.avi"))
    if not video_files: return False
    video_path = video_files[0]
    
    frame_ranges = [(ann["start_frame"], ann["end_frame"]) for ann in annotations.get("annotations", []) if "start_frame" in ann]
    if not frame_ranges: return False
    
    # Extract frames
    if frames_dir.exists():
        print(f"⚠️  Frames directory already exists: {frames_dir} (will reuse existing frames)")
        extracted_frames = sorted(frames_dir.glob("frame_*.jpg"), key=lambda p: int(p.stem.split("_")[1]))
        extracted_frames = [(int(p.stem.split("_")[1]), p) for p in extracted_frames]
    else:
        extracted_frames = extract_frames_from_video(video_path, frame_ranges, frames_dir)
        if not extracted_frames: return False
    
    # ---------------------------------------------------------
    # DYNAMIC MODEL ROUTING
    # ---------------------------------------------------------
    is_olympic = "olympic" in str(dir_path).lower()
    active_model = model_olympic if is_olympic else model_generic
    model_name = "YOLOv11x-pose (Custom Olympic)" if is_olympic else "YOLO11x-pose (Pre-trained Generic)"
    
    print(f"🤖 Routing to: {model_name}")
    pose_annotations = run_yolo_inference(active_model, extracted_frames, is_custom_model=is_olympic)
    
    # Create Output JSON
    yolo_output = {
        "info": {"description": "YOLO 14-point skeleton annotations", "version": "1.0"},
        "categories": [{"id": 1, "name": "person"}],
        "annotations": []
    }
    
    ann_id = 1
    for frame_num in sorted(pose_annotations.keys()):
        for person_ann in pose_annotations[frame_num]:
            annotation = {
                "id": ann_id,
                "frame_number": frame_num,
                "category_id": 1,
                "keypoints": person_ann["keypoints"],
                "bbox": person_ann["bbox"],
                "area": person_ann["area"],
                "num_keypoints": person_ann["num_keypoints"],
                "iscrowd": 0
            }
            yolo_output["annotations"].append(annotation)
            ann_id += 1
            
    with open(yolo_ann_path, 'w') as f:
        json.dump(yolo_output, f, indent=2)
    print(f"✅ Saved YOLO annotations: {yolo_ann_path.name} ({len(yolo_output['annotations'])} poses)")
    
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/DATASET")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    
    data_root = PROJECT_ROOT / args.data_dir
    models_dir = PROJECT_ROOT / args.models_dir
    
    if not data_root.exists():
        print(f"❌ Data directory not found: {data_root}")
        return
        
    print(f"\n{'='*80}\nINITIALIZING MODELS\n{'='*80}\n")
    model_olympic, model_generic = setup_yolo_models(models_dir)
    
    annotation_files = sorted(list(data_root.rglob("annotations.json")))
    if not annotation_files: return
    
    for ann_path in annotation_files:
        process_single_directory(ann_path, model_olympic, model_generic, False)
        
    print(f"\n✅ Processing Complete!")

if __name__ == "__main__":
    main()