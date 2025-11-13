import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Union, Optional
from ultralytics.engine.results import Results
from datetime import datetime


# -------------------------------
# Custom 14-point skeleton mapping
# -------------------------------
KEYPOINT_NAMES = [
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_elbow", "left_wrist",
    "right_elbow", "right_wrist",
    "neck", "nose"
]

SKELETON_CONNECTIONS = [
    [9, 10], [1, 3], [2, 4], [1, 2], [3, 4],
    [6, 8], [4, 6], [5, 7], [11, 12],
    [14, 13], [1, 13], [13, 2], [2, 11],
    [1, 9], [3, 5]
]

# Default output path relative to this script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "annotations" / "test.json"


def to_custom_keypoints(yolo_kps):
    """
    Accepts either:
      - 14 x (2|3) : already in custom order (nose, neck, left_shoulder, ...)
      - 17 x (2|3) : COCO order -> convert to custom 14
    Returns: np.array shape (14,2) or (14,3) or None if unsupported
    """
    if yolo_kps is None:
        return None
    if yolo_kps.ndim != 2:
        print(f"[DEBUG] to_custom_keypoints: unexpected dims {yolo_kps.shape}")
        return None

    n_kps, dim = yolo_kps.shape
    has_conf = (dim == 3)

    # If already 14 keypoints — assume correct custom order
    if n_kps == 14:
        return yolo_kps.copy()

    # If 17 -> map COCO -> custom 14 (same mapping as you intended)
    if n_kps == 17:
        left_shoulder, right_shoulder = yolo_kps[5], yolo_kps[6]
        if has_conf:
            neck = np.array([
                (left_shoulder[0] + right_shoulder[0]) / 2.0,
                (left_shoulder[1] + right_shoulder[1]) / 2.0,
                float(min(left_shoulder[2], right_shoulder[2]))
            ])
        else:
            neck = (left_shoulder + right_shoulder) / 2.0

        custom = [
            yolo_kps[0],     # nose (0)
            neck,            # neck (computed)
            yolo_kps[5],     # left_shoulder
            yolo_kps[6],     # right_shoulder
            yolo_kps[7],     # left_elbow
            yolo_kps[8],     # right_elbow
            yolo_kps[9],     # left_wrist
            yolo_kps[10],    # right_wrist
            yolo_kps[11],    # left_hip
            yolo_kps[12],    # right_hip
            yolo_kps[13],    # left_knee
            yolo_kps[14],    # right_knee
            yolo_kps[15],    # left_ankle
            yolo_kps[16]     # right_ankle
        ]
        return np.array(custom)

    # unsupported count
    print(f"[DEBUG] to_custom_keypoints: unsupported keypoint count = {n_kps}")
    return None


def save_results_as_coco(
    results: List[Results],
    image_paths: List[Path],
    out_json: Optional[Path] = None,
    overwrite: bool = True
):
    """Convert YOLO pose results to COCO format with proper class tracking."""
    
    if out_json is None:
        out_json = DEFAULT_OUTPUT_PATH
        print(f"[INFO] Using default output path: {out_json}")
    
    out_json = Path(out_json)
    
    if out_json.exists() and not overwrite:
        raise FileExistsError(f"Output file exists: {out_json}")
    
    out_json.parent.mkdir(parents=True, exist_ok=True)
    
    # COCO structure
    info = {
        "description": "YOLO Pose with RF-DETR Boxing Classes",
        "version": "1.0",
        "year": datetime.now().year,
        "contributor": "YOLO + RF-DETR Pipeline",
        "date_created": datetime.now().strftime("%Y/%m/%d")
    }
    
    licenses = [{"id": 1, "name": "Unknown", "url": ""}]
    
    images_info = []
    annotations = []
    ann_id = 1
    
    # Define categories for BOTH boxer types
    categories = [
        {
            "id": 0,
            "name": "boxer_blue",
            "supercategory": "person",
            "keypoints": KEYPOINT_NAMES,
            "skeleton": SKELETON_CONNECTIONS
        },
        {
            "id": 1,
            "name": "boxer_red",
            "supercategory": "person",
            "keypoints": KEYPOINT_NAMES,
            "skeleton": SKELETON_CONNECTIONS
        }
    ]
    
    # Process each image
    for img_id, (result, img_path) in enumerate(zip(results, image_paths), start=1):
        h, w = result.orig_shape
        
        images_info.append({
            "id": img_id,
            "file_name": img_path.name,
            "height": int(h),
            "width": int(w),
            "license": 1,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": ""
        })

        # Get class information from RF-DETR
        class_names = getattr(result, 'class_names', [])
        
        if not hasattr(result, 'keypoints') or result.keypoints is None:
            print(f"[DEBUG] Image {img_id} ({img_path.name}): No keypoints")
            continue
        
        keypoints = result.keypoints.xy
        kp_conf = None
        if hasattr(result.keypoints, 'conf') and result.keypoints.conf is not None:
            kp_conf = result.keypoints.conf
        
        bboxes = None
        if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
            bboxes = result.boxes.xyxy
        
        # Convert to numpy
        keypoints_np = keypoints.cpu().numpy()
        kp_conf_np = kp_conf.cpu().numpy() if kp_conf is not None else None
        bboxes_np = bboxes.cpu().numpy() if bboxes is not None else None
        
        # Stack with confidence
        if kp_conf_np is not None:
            keypoints_with_conf = np.concatenate([
                keypoints_np, 
                kp_conf_np[..., np.newaxis]
            ], axis=-1)
        else:
            keypoints_with_conf = keypoints_np
        
        # Process each detected person
        for person_idx in range(keypoints_np.shape[0]):
            # Determine category from RF-DETR class names
            category_id = 0  # default to boxer_blue
            
            if class_names and person_idx < len(class_names):
                class_name = class_names[person_idx]
                category_id = 1 if class_name == "boxer_red" else 0
                print(f"[DEBUG] Image {img_id}, Person {person_idx}: {class_name} -> cat {category_id}")
            else:
                print(f"[WARNING] Image {img_id}, Person {person_idx}: No class, using blue")
            
            # Convert keypoints once
            kp = to_custom_keypoints(keypoints_with_conf[person_idx])
            if kp is None:
                print(f"[DEBUG] Image {img_id}, Person {person_idx}: Keypoint conversion failed")
                continue
            
            # Format keypoints as [x1, y1, v1, x2, y2, v2, ...]
            kp_with_vis = []
            num_visible = 0
            
            for point in kp:
                if point.shape[0] == 3:
                    x, y, conf = point
                    # More lenient visibility threshold
                    if conf > 0.3:
                        v = 2
                        num_visible += 1
                    elif conf > 0:
                        v = 1
                    else:
                        v = 0
                else:
                    x, y = point
                    if x > 0 and y > 0:
                        v = 2
                        num_visible += 1
                    else:
                        v = 0
                
                kp_with_vis.extend([float(x), float(y), int(v)])
            
            # Keep annotations with at least 3 visible keypoints
            if num_visible < 3:
                print(f"[DEBUG] Image {img_id}, Person {person_idx}: Only {num_visible} visible")
                continue
            
            # Calculate bbox
            if bboxes_np is not None and person_idx < len(bboxes_np):
                x1, y1, x2, y2 = bboxes_np[person_idx]
                bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            else:
                kp_xy = kp[:, :2] if kp.shape[1] >= 2 else kp
                xs, ys = kp_xy[:, 0], kp_xy[:, 1]
                valid_mask = (xs > 0) & (ys > 0)
                if valid_mask.sum() == 0:
                    continue
                xs_valid, ys_valid = xs[valid_mask], ys[valid_mask]
                bbox = [
                    float(xs_valid.min()), 
                    float(ys_valid.min()),
                    float(xs_valid.max() - xs_valid.min()), 
                    float(ys_valid.max() - ys_valid.min())
                ]
            
            # Create annotation with correct category
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": category_id,
                "keypoints": kp_with_vis,
                "num_keypoints": num_visible,
                "bbox": bbox,
                "iscrowd": 0,
                "area": float(bbox[2] * bbox[3]),
                "segmentation": []
            })
            ann_id += 1
    
    coco_data = {
        "info": info,
        "licenses": licenses,
        "images": images_info,
        "annotations": annotations,
        "categories": categories
    }
    
    with open(out_json, "w") as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"\n[INFO] ✅ COCO annotations saved to: {out_json}")
    print(f"[INFO] Total images: {len(images_info)}")
    print(f"[INFO] Total annotations: {len(annotations)}")
    
    # Print category breakdown
    blue_count = sum(1 for ann in annotations if ann['category_id'] == 0)
    red_count = sum(1 for ann in annotations if ann['category_id'] == 1)
    print(f"[INFO] Boxer Blue: {blue_count}, Boxer Red: {red_count}")