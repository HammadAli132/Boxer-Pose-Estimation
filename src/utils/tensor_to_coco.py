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

def heatmap_to_keypoints(heatmaps: torch.Tensor, crop_size: int, confidence_threshold: float = 0) -> List[float]:
    """
    Converts a single (14, H_out, W_out) heatmap tensor to 14 (x, y, v) tuples.
    This assumes the input heatmap is scaled relative to the *cropped* input image.
    Returns a flat list of [x1, y1, v1, x2, y2, v2, ...] in crop coordinates.
    """
    kps_list = []
    num_kps, H_out, W_out = heatmaps.shape
    
    # Scaling factor: scale from heatmap resolution to crop size
    scale_x = crop_size / W_out
    scale_y = crop_size / H_out
    
    for k in range(num_kps):
        kp_map = heatmaps[k]
        # Find maximum value and its location
        max_val = kp_map.max()
        
        if max_val.item() < confidence_threshold:
            kps_list.extend([0.0, 0.0, 0])  # Not visible (v=0)
            continue
            
        y_out, x_out = torch.where(kp_map == max_val)
        x_out, y_out = x_out[0].item(), y_out[0].item()
        
        # Convert from heatmap coordinates to crop coordinates
        x_crop = x_out * scale_x 
        y_crop = y_out * scale_y
        
        # Confidence determines visibility (v=2 for visible)
        v = 2 
        kps_list.extend([x_crop, y_crop, v])
        
    return kps_list


def tensor_to_coco_json(
    results: List[Dict[str, Any]],  # List of result dictionaries
    image_info: List[Dict[str, Any]],
    out_json: Path = DEFAULT_OUTPUT_PATH,
    crop_size: int = 224,
    overwrite: bool = True
):
    """
    Converts list of result dictionaries into COCO JSON format.
    
    Each result dict should contain:
        - heatmaps: torch.Tensor of shape (1, NUM_KEYPOINTS, H_out, W_out)
        - bbox_xywh: [x, y, w, h] in frame coordinates
        - image_id: int
        - annotation_id: int
        - class_name: str (e.g., 'boxer_blue', 'boxer_red')
        - track_id: int or str
        - bbox_xyxy: [x1, y1, x2, y2] in frame coordinates
    """
    if out_json.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {out_json}")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize COCO structure
    coco_data = {
        "info": {
            "description": "DinoV2 Keypoint Detection Results",
            "date_created": datetime.now().isoformat()
        },
        "licenses": [],
        "categories": [{
            "id": 1,
            "name": "boxer",
            "supercategory": "person",
            "keypoints": KEYPOINT_NAMES,
            "skeleton": SKELETON_CONNECTIONS
        }],
        "images": image_info,
        "annotations": []
    }
    
    # Process each detection result
    for result in results:
        heatmaps = result['heatmaps']  # (1, NUM_KEYPOINTS, H_out, W_out)
        bbox_xywh = result['bbox_xywh']  # [x, y, w, h]
        image_id = result['image_id']
        annotation_id = result['annotation_id']
        
        # Extract bbox coordinates
        x_min, y_min, w, h = bbox_xywh
        
        # 1. Convert heatmaps to (x, y, v) list in crop coordinates
        kp_crop_list = heatmap_to_keypoints(heatmaps.squeeze(0), crop_size=crop_size)
        
        # 2. Transform keypoints from crop coordinates to frame coordinates
        kp_frame_list = []
        num_visible = 0
        
        for i in range(0, len(kp_crop_list), 3):
            x_crop, y_crop, v = kp_crop_list[i:i+3]
            
            if v > 0:
                # Scale keypoint from crop space (0 to crop_size) to bbox space (0 to w/h)
                # Then translate to frame coordinates
                x_frame = (x_crop / crop_size) * w + x_min
                y_frame = (y_crop / crop_size) * h + y_min
                kp_frame_list.extend([x_frame, y_frame, v])
                
                if v == 2:
                    num_visible += 1
            else:
                kp_frame_list.extend([0.0, 0.0, 0])
        
        # Calculate area
        area = float(w * h)
        
        # Create annotation entry
        coco_data["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,
            "keypoints": kp_frame_list,
            "num_keypoints": num_visible,
            "bbox": [float(x_min), float(y_min), float(w), float(h)],
            "iscrowd": 0,
            "area": area,
            "segmentation": [],
            # Optional: store additional metadata
            "track_id": result.get('track_id'),
            "class_name": result.get('class_name')
        })
    
    # Write to JSON file
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, indent=2)

    print(f"✅ COCO annotations saved to: {out_json}")
    print(f"   Total images: {len(coco_data['images'])}")
    print(f"   Total annotations: {len(coco_data['annotations'])}")