import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
from datetime import datetime

# Define output path (shared with inference.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "annotations" / "test.json"

# Define skeleton (must match DinoV2 pipeline definition)
KEYPOINT_NAMES = [
    "left_shoulder", "right_shoulder", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_elbow", "left_wrist", "right_elbow", "right_wrist",
    "neck", "nose"
]
SKELETON_CONNECTIONS = [
    [9, 10], [1, 3], [2, 4], [1, 2], [3, 4],
    [6, 8], [4, 6], [5, 7], [11, 12],
    [14, 13], [1, 13], [13, 2], [2, 11],
    [1, 9], [3, 5]
]

def heatmap_to_keypoints(heatmaps: torch.Tensor, confidence_threshold: float = 0.5) -> List[Tuple[float, float, int]]:
    """
    Converts a single (14, H_out, W_out) heatmap tensor to 14 (x, y, v) tuples.
    This assumes the input heatmap is scaled relative to the *cropped* input image.
    """
    kps_list = []
    num_kps, H_out, W_out = heatmaps.shape
    
    # 1. Scaling factor: assume final heatmap resolution is a multiple of input image resolution
    # For simplicity, we assume we need to scale the predicted coordinates (from heatmap space)
    # back to the expected input crop size (e.g., 256x256 crop). 
    # NOTE: This factor must be calibrated with the input size and upsampling in the head.
    scale_factor = 640 / W_out # Assuming 640 input and W_out=160 (4x upsample)
    
    for k in range(num_kps):
        kp_map = heatmaps[k]
        # Find maximum value and its location (y, x in array terms)
        max_val = kp_map.max()
        if max_val.item() < confidence_threshold:
            kps_list.extend([0.0, 0.0, 0]) # Not visible (v=0)
            continue
            
        y_out, x_out = torch.where(kp_map == max_val)
        x_out, y_out = x_out[0].item(), y_out[0].item()
        
        # 2. Convert from heatmap coordinates to original crop coordinates
        x_crop = x_out * scale_factor 
        y_crop = y_out * scale_factor
        
        # Confidence determines visibility (v=2 for visible)
        v = 2 
        kps_list.extend([x_crop, y_crop, v])
        
    return kps_list

def tensor_to_coco_json(
    results: List[Tuple[torch.Tensor, Tuple[float, float, float, float]]], # (Heatmaps, BBox_in_Frame)
    image_info: List[Dict[str, Any]],
    out_json: Path = DEFAULT_OUTPUT_PATH,
    overwrite: bool = True
):
    """
    Converts list of (Heatmap Tensor, BBox) results into COCO JSON.
    """
    if out_json.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {out_json}")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize COCO structure (simplified)
    coco_data = {
        "info": {"description": "DinoV2 Heatmap to COCO Conversion"},
        "licenses": [],
        "categories": [{
            "id": 1,
            "name": "boxer",
            "supercategory": "person",
            "keypoints": KEYPOINT_NAMES,
            "skeleton": SKELETON_CONNECTIONS
        }],
        "images": image_info, # Image metadata passed from inference script
        "annotations": []
    }
    
    ann_id = 1
    
    # NOTE: Since TF-DETR (Detector) handles the multi-person part, 
    # each item in 'results' corresponds to a single boxer instance.
    
    for result_idx, (heatmaps, bbox_frame) in enumerate(results):
        # 1. Convert heatmaps to (x, y, v) list in crop coordinates
        kp_crop_list = heatmap_to_keypoints(heatmaps.squeeze(0)) # Squeeze B dim
        
        # 2. Convert crop coordinates to full frame coordinates
        x_min, y_min, x_max, y_max = bbox_frame # BBox in frame coordinates
        
        # Calculate BBox for COCO format: [x, y, w, h]
        bbox_coco = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]
        area = bbox_coco[2] * bbox_coco[3]
        
        kp_frame_list = []
        num_visible = 0
        
        for i in range(0, len(kp_crop_list), 3):
            x_crop, y_crop, v = kp_crop_list[i:i+3]
            
            # Keypoint is in the frame
            x_frame = x_crop + x_min
            y_frame = y_crop + y_min
            
            kp_frame_list.extend([x_frame, y_frame, v])
            if v == 2:
                num_visible += 1

        # Determine which image this annotation belongs to (simplified)
        # This requires tracking the original image ID/name from the detector step.
        # For simplicity in this skeleton, we assume `image_info` tracks this.
        image_id = result_idx + 1 # Needs actual mapping 
        
        coco_data["annotations"].append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": 1,
            "keypoints": kp_frame_list,
            "num_keypoints": num_visible,
            "bbox": bbox_coco,
            "iscrowd": 0,
            "area": float(area),
            "segmentation": []
        })
        ann_id += 1
        
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, indent=2)

    print(f"[INFO] COCO annotations saved to: {out_json}")
    print(f"[INFO] Total annotations converted: {len(coco_data['annotations'])}")