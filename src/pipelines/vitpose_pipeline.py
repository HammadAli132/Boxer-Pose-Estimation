#!/usr/bin/env python3
"""
vitpose_pipeline.py — inference-only pipeline using MMPose ViT-Pose (top-down)

Usage:
  python src/pipelines/vitpose_pipeline.py --config /path/to/config.py --checkpoint /path/to/checkpoint.pth

Behavior:
  - For each image in data/processed/images/test:
      - If /data/processed/annotations/rf_detr/{image_stem}.json exists, use its detections (bbox_xyxy) as person boxes.
      - Otherwise use a fallback bbox that covers the whole image.
  - For each bbox, run inference_top_down_pose_model (MMPose).
  - Convert pose outputs to COCO keypoints 1.0-like JSON (custom 14-key mapping).
  - Save output to data/processed/annotations/test.json (overwrites).
"""

import json
import os
from pathlib import Path
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
import requests
from tqdm import tqdm
import numpy as np
from PIL import Image

# mmpose imports
try:
    from mmpose.apis import init_model, inference_top_down_pose_model
except Exception as e:
    raise ImportError("mmpose is required for this script. Install mmpose and mmcv and retry.") from e

# ---- Project paths ----
PROJECT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = PROJECT / "data" / "processed"
IMAGES_TEST = DATA_PROCESSED / "images" / "test"
ANNOTS_PROCESSED = DATA_PROCESSED / "annotations"
OUT_TEST_JSON = ANNOTS_PROCESSED / "test.json"
RFDETR_ANNOTS_DIR = ANNOTS_PROCESSED / "rf_detr"

# ---- Custom 14-key configuration (your mapping) ----
KEYPOINT_NAMES = [
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_elbow", "left_wrist",
    "right_elbow", "right_wrist",
    "neck", "nose"
]

SKELETON = [
    [9, 10], [1, 3], [2, 4], [1, 2], [3, 4],
    [6, 8], [4, 6], [5, 7], [11, 12],
    [14, 13], [1, 13], [13, 2], [2, 11],
    [1, 9], [3, 5]
]

# Helper: convert mmpose (likely COCO/17) keypoints to custom 14
def to_custom_keypoints(yolo_kps: np.ndarray) -> Optional[np.ndarray]:
    """
    Accepts array of shape (17,2) or (17,3) or (14,2)/(14,3).
    Returns array (14,2) or (14,3) in our custom order:
      [nose, neck, left_shoulder, right_shoulder, left_elbow, right_elbow,
       left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee,
       left_ankle, right_ankle]
    """
    if yolo_kps is None:
        return None
    y = np.asarray(yolo_kps)
    if y.ndim != 2:
        return None
    n, d = y.shape
    has_conf = (d == 3)

    if n == 14:
        return y.copy()

    if n == 17:
        # indices in COCO: 0:nose, 5:left_shoulder,6:right_shoulder, 7:left_elbow,8:right_elbow,
        # 9:left_wrist,10:right_wrist,11:left_hip,12:right_hip,13:left_knee,14:right_knee,15:left_ankle,16:right_ankle
        left_shoulder = y[5]
        right_shoulder = y[6]
        if has_conf:
            neck = np.array([
                (left_shoulder[0] + right_shoulder[0]) / 2.0,
                (left_shoulder[1] + right_shoulder[1]) / 2.0,
                float(min(left_shoulder[2], right_shoulder[2]))
            ])
        else:
            neck = (left_shoulder + right_shoulder) / 2.0

        custom = [
            y[0],     # nose
            neck,     # neck
            y[5],     # left_shoulder
            y[6],     # right_shoulder
            y[7],     # left_elbow
            y[8],     # right_elbow
            y[9],     # left_wrist
            y[10],    # right_wrist
            y[11],    # left_hip
            y[12],    # right_hip
            y[13],    # left_knee
            y[14],    # right_knee
            y[15],    # left_ankle
            y[16],    # right_ankle
        ]
        return np.array(custom, dtype=float)

    # unsupported shape
    return None


def build_coco_structure(images_info: List[Dict[str, Any]], annotations: List[Dict[str, Any]]):
    info = {
        "description": "ViT-Pose automatic predictions",
        "url": "",
        "version": "1.0",
        "year": datetime.now().year,
        "contributor": "vitpose_pipeline",
        "date_created": datetime.now().strftime("%Y/%m/%d")
    }
    licenses = [{"id": 1, "name": "", "url": ""}]
    categories = [
        {"id": 1, "name": "quadruped", "supercategory": "", "keypoints": [], "skeleton": []},
        {"id": 2, "name": "person", "supercategory": "", "keypoints": KEYPOINT_NAMES, "skeleton": SKELETON}
    ]
    return {
        "info": info,
        "licenses": licenses,
        "images": images_info,
        "annotations": annotations,
        "categories": categories
    }


def load_rf_detr_boxes(img_stem: str) -> List[Dict[str, Any]]:
    path = RFDETR_ANNOTS_DIR / f"{img_stem}.json"
    if not path.exists():
        return []
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d.get("detections", []) or d.get("boxes", []) or []
    except Exception:
        return []

def ensure_vitpose_model(model_name: str = "vitpose-base-coco-256x192"):
    """
    Ensures that the ViT-Pose model (config + checkpoint) exists locally.
    If missing, downloads from MMPose model zoo into PROJECT/models/{model_name}/
    Returns (config_path, checkpoint_path)
    """

    MODEL_DIR = PROJECT / "models" / model_name
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    config_path = MODEL_DIR / "config.py"
    ckpt_path = MODEL_DIR / "model.pth"

    # official URLs (can be updated for other variants)
    MODEL_URLS = {
        "vitpose-base-coco-256x192": {
            "config": "https://github.com/open-mmlab/mmpose/blob/main/configs/body_2d_keypoint/topdown_heatmap/coco/vitpose/vitpose-base-coco-256x192.py?raw=true",
            "ckpt": "https://download.openmmlab.com/mmpose/top_down/vitpose/vitpose-base-coco-256x192-8ac66bf9_20230314.pth"
        },
        "vitpose-huge-coco-384x288": {
            "config": "https://github.com/open-mmlab/mmpose/blob/main/configs/body_2d_keypoint/topdown_heatmap/coco/vitpose/vitpose-huge-coco-384x288.py?raw=true",
            "ckpt": "https://download.openmmlab.com/mmpose/top_down/vitpose/vitpose-huge-coco-384x288-fb7d5c5e_20230314.pth"
        }
    }

    if model_name not in MODEL_URLS:
        raise ValueError(f"Unknown model_name '{model_name}'. Available: {list(MODEL_URLS.keys())}")

    urls = MODEL_URLS[model_name]

    # helper to download with progress
    def download_file(url, dest):
        if dest.exists():
            print(f"[INFO] Found existing: {dest.name}")
            return
        print(f"[INFO] Downloading {dest.name} ...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in tqdm(r.iter_content(chunk_size=8192), total=None):
                    if chunk:
                        f.write(chunk)

    # download both if missing
    download_file(urls["config"], config_path)
    download_file(urls["ckpt"], ckpt_path)

    print(f"[INFO] Model ready at: {MODEL_DIR}")
    return config_path, ckpt_path

def topdown_inference_for_image(model, img_path: Path, device: str, bbox_list: List[Dict[str, Any]], bbox_score_thr=0.0):
    """
    Returns list of pose results. Each pose_result is dict with 'keypoints' and 'bbox' and 'score'.
    Uses mmpose.inference_top_down_pose_model
    """
    img = str(img_path)
    # Build person_results from bbox_list
    person_results = []
    for det in bbox_list:
        # det expected to have 'bbox_xyxy' or 'bbox'
        if "bbox_xyxy" in det:
            x1, y1, x2, y2 = det["bbox_xyxy"]
        elif "bbox" in det:
            x1, y1, w, h = det["bbox"]
            x2, y2 = x1 + w, y1 + h
        else:
            continue
        person_results.append({"bbox": [float(x1), float(y1), float(x2), float(y2)]})

    # If no bbox_list provided, fallback to full-image bbox
    if not person_results:
        from PIL import Image
        pil = Image.open(img_path)
        W, H = pil.size
        person_results = [{"bbox": [0.0, 0.0, float(W - 1), float(H - 1)]}]

    # Run top-down pose inference
    # returns a list of lists: pose_results (list of dicts) and returned_scores
    pose_results, returned_scores = inference_top_down_pose_model(
        model, img, person_results, bbox_thr=bbox_score_thr, format='xyxy', output_heatmap=False
    )
    # Each element in pose_results is dict {'bbox':..., 'keypoints': np.array(...)}
    return pose_results


def run_vitpose_inference(
    config_path: str,
    checkpoint_path: str,
    device: Optional[str] = None,
    score_thr: float = 0.3
) -> None:
    # pick device
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    print(f"[INFO] Initializing ViT-Pose model from config:{config_path} checkpoint:{checkpoint_path}")
    model = init_model(config_path, checkpoint_path, device=device)

    # Gather test images
    imgs = sorted([p for p in IMAGES_TEST.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".png"]])
    if not imgs:
        print(f"[WARN] No test images found in {IMAGES_TEST}")
        return

    images_info = []
    annotations = []
    ann_id = 1

    print(f"[INFO] Running inference on {len(imgs)} images...")

    for img_idx, img_path in enumerate(imgs, start=1):
        img_stem = img_path.stem
        pil = Image.open(img_path)
        W, H = pil.size

        images_info.append({
            "id": img_idx,
            "width": W,
            "height": H,
            "file_name": img_path.name,
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": 0
        })

        # load RF-DETR boxes for this image
        dets = load_rf_detr_boxes(img_stem)

        # convert to person_results and do top-down inference
        try:
            pose_results = topdown_inference_for_image(model, img_path, device, dets, bbox_score_thr=0.0)
        except Exception as e:
            print(f"[WARN] Pose inference failed for {img_path.name}: {e}")
            continue

        # pose_results is a list of dicts; each dict should have 'keypoints' np array (K,2) or (K,3)
        for pr in pose_results:
            if "keypoints" not in pr or pr["keypoints"] is None:
                continue
            kps = np.array(pr["keypoints"], dtype=float)  # shape: (K,2) or (K,3)
            # convert to custom 14
            custom = to_custom_keypoints(kps)
            if custom is None:
                # skip if mapping failed
                continue

            # build keypoints list [x,y,v] where v: 2 visible (conf>0.5), 1 labeled but not visible, 0 not labeled
            kp_list = []
            num_visible = 0
            for point in custom:
                if point.shape[0] == 3:
                    x, y, conf = float(point[0]), float(point[1]), float(point[2])
                    v = 2 if conf > 0.5 else 1 if conf > 0.0 else 0
                    if v == 2:
                        num_visible += 1
                else:
                    x, y = float(point[0]), float(point[1])
                    v = 2 if (x > 0 and y > 0) else 0
                    if v == 2:
                        num_visible += 1
                kp_list.extend([x, y, int(v)])

            # bbox: prefer pr['bbox'] if present
            if "bbox" in pr and pr["bbox"] is not None:
                bx1, by1, bx2, by2 = pr["bbox"]
                bw = float(bx2 - bx1)
                bh = float(by2 - by1)
                bbox = [float(bx1), float(by1), bw, bh]
            else:
                # compute from keypoints
                xs = np.array(kp_list[0::3])
                ys = np.array(kp_list[1::3])
                valid = (xs > 0) & (ys > 0)
                if valid.sum() == 0:
                    continue
                xmin, xmax = float(xs[valid].min()), float(xs[valid].max())
                ymin, ymax = float(ys[valid].min()), float(ys[valid].max())
                bbox = [xmin, ymin, xmax - xmin, ymax - ymin]

            area = bbox[2] * bbox[3]
            annotations.append({
                "id": ann_id,
                "image_id": img_idx,
                "category_id": 2,   # person id in your categories
                "segmentation": [],
                "area": float(area),
                "bbox": [float(b) for b in bbox],
                "iscrowd": 0,
                "attributes": {},
                "keypoints": [float(x) if not isinstance(x, np.generic) else float(np.float64(x)) for x in kp_list],
                "num_keypoints": int(num_visible)
            })
            ann_id += 1

        if img_idx % 50 == 0:
            print(f"[INFO] Processed {img_idx}/{len(imgs)} images...")

    # write COCO json
    OUT_TEST_JSON.parent.mkdir(parents=True, exist_ok=True)
    coco = build_coco_structure(images_info, annotations)
    with open(OUT_TEST_JSON, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)

    print(f"[INFO] Saved COCO keypoints JSON to: {OUT_TEST_JSON}")
    print(f"[INFO] Images: {len(images_info)}, Annotations: {len(annotations)}")


def main():
    parser = argparse.ArgumentParser(description="ViT-Pose inference pipeline (top-down using RF-DETR crops)")
    parser.add_argument("--config", required=True, help="MMPose config (.py) for ViT-Pose")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint (.pth) for ViT-Pose")
    parser.add_argument("--device", default='cuda:0', help="device string, e.g. 'cuda:0' or 'cpu' (auto if empty)")
    parser.add_argument("--score-thr", type=float, default=0.3, help="keypoint confidence threshold for visibility mapping")
    args = parser.parse_args()

    if args.config.lower() == "auto" or args.checkpoint.lower() == "auto":
        config_path, checkpoint_path = ensure_vitpose_model("vitpose-base-coco-384x288")
    else:
        config_path, checkpoint_path = args.config, args.checkpoint

    run_vitpose_inference(config_path, checkpoint_path, device=args.device, score_thr=args.score_thr)


if __name__ == "__main__":
    main()
