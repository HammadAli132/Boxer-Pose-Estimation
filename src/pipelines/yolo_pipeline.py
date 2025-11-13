"""
src/pipelines/yolo_pipeline.py

Enhanced YOLO Pose training script:
- Prompts user to recreate labels (COCO → YOLO) each run.
- Saves model weights to /models/{model_name}/ (best.pt, last.pt).
- Saves training results (graphs, metrics, etc.) to /runs/run_{n}/.
"""

import os
import sys
import shutil
import argparse
import time
import traceback
import torch
from pathlib import Path
import requests
from ultralytics import YOLO
import yaml
from ultralytics.engine.results import Results
from torchvision import transforms as T
from PIL import Image
import json
import numpy as np
from tkinter import Tk, filedialog
import cv2

# Import your converter
from ..utils.coco_to_yolo_pose import coco_to_yolo_keypoints
from ..utils.results_to_coco import save_results_as_coco

# -----------------------
# Project paths
# -----------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
IMAGES_DIR = DATA_PROCESSED / "images"
ANNOTS_DIR = DATA_PROCESSED / "annotations"
LABELS_DIR = DATA_PROCESSED / "labels"
MODELS_ROOT = PROJECT_ROOT / "models"
IMAGES_TEST = DATA_PROCESSED / "images" / "test"
OUT_TEST_JSON = ANNOTS_DIR / "test.json"
RFDETR_ANNOTS_DIR = DATA_PROCESSED / "annotations/rf-detr"
RUNS_ROOT = PROJECT_ROOT / "runs"  # <-- NEW: store training results here

# -----------------------
# Defaults
# -----------------------
DEFAULT_MODEL = "yolov8m-pose"
NUM_KEYPOINTS = 14
EPOCHS = 50
IMGSZ = 640
DEFAULT_BATCH = 8
DEFAULT_WORKERS = 0

WEIGHTS_MAP = {
    "yolov8s-pose": "yolov8s-pose.pt",
    "yolov8m-pose": "yolov8m-pose.pt",
    "yolov11-pose": "yolov11-pose.pt",
}

KEYPOINTS = [
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


# -----------------------
# Helpers
# -----------------------
def download_file(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def ensure_pretrained(model_name: str) -> Path:
    """Ensure pretrained weights are present and return local path."""
    if model_name not in WEIGHTS_MAP:
        raise ValueError(f"Unknown model_name: {model_name}")

    model_dir = MODELS_ROOT / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    local_file = model_dir / WEIGHTS_MAP[model_name]

    if local_file.exists():
        return local_file

    url = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{WEIGHTS_MAP[model_name]}"
    print(f"⬇️ Downloading pretrained weights for {model_name} to {local_file} ...")
    download_file(url, local_file)
    print("✅ Download complete.")
    return local_file


def dataset_yaml_path_for(model_name: str) -> Path:
    yaml_path = PROJECT_ROOT / f"{model_name}_dataset.yaml"
    content = {
        "path": str(DATA_PROCESSED),
        "train": str(IMAGES_DIR / "train"),
        "val": str(IMAGES_DIR / "val"),
        "kpt_shape": [NUM_KEYPOINTS, 3],
        "names": ["boxer_red", "boxer_blue"],
        "keypoints": KEYPOINTS,
        "skeleton": SKELETON,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(content, f)
    return yaml_path


def labels_exist(split: str) -> bool:
    dirp = LABELS_DIR / split
    return dirp.exists() and any(dirp.glob("*.txt"))


def ensure_labels_from_coco(split: str, coco_json: Path, images_dir: Path, out_labels_dir: Path, num_kpts: int):
    if out_labels_dir.exists():
        shutil.rmtree(out_labels_dir)
    out_labels_dir.mkdir(parents=True, exist_ok=True)
    print(f"🔁 Converting {coco_json} -> {out_labels_dir} ...")
    coco_to_yolo_keypoints(str(coco_json), str(images_dir), str(out_labels_dir), num_keypoints=num_kpts)
    print("✅ Conversion done.")


def ask_yes_no(prompt: str) -> bool:
    """Utility for yes/no questions in CLI."""
    while True:
        ans = input(f"{prompt} [y/n]: ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("Please enter 'y' or 'n'.")


# -----------------------
# Training Logic
# -----------------------
def run_yolo_training(
    model_name: str = DEFAULT_MODEL,
    epochs: int = EPOCHS,
    imgsz: int = IMGSZ,
    batch: int = DEFAULT_BATCH,
    workers: int = DEFAULT_WORKERS,
    resume_from_best: bool = True,
):
    # Paths setup
    model_dir = MODELS_ROOT / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    # Annotations
    train_json = ANNOTS_DIR / "train.json"
    val_json = ANNOTS_DIR / "val.json"
    if not train_json.exists() or not val_json.exists():
        raise FileNotFoundError("Train/Val JSONs not found.")

    # Ask if user wants to recreate labels
    recreate_labels = ask_yes_no("Do you want to recreate YOLO labels from COCO annotations?")
    if recreate_labels or not (labels_exist("train") and labels_exist("val")):
        ensure_labels_from_coco("train", train_json, IMAGES_DIR / "train", LABELS_DIR / "train", NUM_KEYPOINTS)
        ensure_labels_from_coco("val", val_json, IMAGES_DIR / "val", LABELS_DIR / "val", NUM_KEYPOINTS)
    else:
        print("✅ Using existing YOLO labels.")

    # Check again
    if not (labels_exist("train") and labels_exist("val")):
        print("❌ ERROR: Labels missing even after conversion.")
        return False

    # Prepare weights & dataset yaml
    local_weights = ensure_pretrained(model_name)
    data_yaml = dataset_yaml_path_for(model_name)
    checkpoint_target = model_dir / "best.pt"

    if resume_from_best and checkpoint_target.exists():
        print(f"🔁 Resuming from checkpoint: {checkpoint_target}")
        weights_to_load = str(checkpoint_target)
    else:
        weights_to_load = str(local_weights)
        print(f"📦 Using pretrained weights: {weights_to_load}")

    # Create YOLO model
    os.chdir(PROJECT_ROOT)
    model = YOLO(weights_to_load)

    # Create new run directory
    run_number = 1
    while (RUNS_ROOT / model_name / f"run_{run_number}").exists():
        run_number += 1
    current_run_dir = RUNS_ROOT / model_name / f"run_{run_number}"
    current_run_dir.mkdir(parents=True)
    print(f"🧾 Logging this training to: {current_run_dir}")

    # Training retries
    attempt = 0
    max_attempts = 3
    current_batch = batch
    current_workers = workers

    while attempt < max_attempts:
        attempt += 1
        try:
            print(f"\n🎯 Training attempt {attempt} — batch={current_batch}, workers={current_workers}")
            results = model.train(
                data=str(data_yaml),
                epochs=epochs,
                imgsz=imgsz,
                batch=current_batch,
                workers=current_workers,
                project=str(current_run_dir),  # now goes to /runs/model_name/run_{n}
                name="",  # avoid subfolder (no /exp)
                exist_ok=True,
                device=0 if torch.cuda.is_available() else "cpu"
            )
                # --- Print validation summary ---
            if isinstance(results, dict):
                print("\n📊 Validation summary:")
                print(results)
            
        except Exception as e:
            tb = traceback.format_exc()
            print(f"\n⚠️ Training failed on attempt {attempt}: {e}\n")
            if "out of memory" in tb.lower() or "WinError 1455" in tb:
                if current_batch > 1:
                    current_batch //= 2
                if current_workers > 0:
                    current_workers -= 1
                print(f"🔧 Retrying with batch={current_batch}, workers={current_workers} ...")
                time.sleep(2)
                continue
            else:
                print("❌ Non-memory error encountered. Aborting.")
                print(tb)
                raise

    # --- Copy best and last weights ---
    exp_weights_dir = current_run_dir / "train" / "weights"  # <-- updated path
    best_src = exp_weights_dir / "best.pt"
    last_src = exp_weights_dir / "last.pt"

    if best_src.exists():
        shutil.copy2(best_src, model_dir / "best.pt")
        print(f"✅ Copied best.pt -> {model_dir / 'best.pt'}")
    else:
        print("⚠️ best.pt not found in run folder.")

    if last_src.exists():
        shutil.copy2(last_src, model_dir / "last.pt")
        print(f"✅ Copied last.pt -> {model_dir / 'last.pt'}")
    else:
        print("⚠️ last.pt not found in run folder.")

    # --- Keep only last 5 runs ---
    runs = sorted(
        [p for p in RUNS_ROOT.glob("run_*") if p.is_dir()],
        key=lambda x: x.stat().st_mtime
    )
    if len(runs) > 5:
        old_runs = runs[:-5]
        for r in old_runs:
            try:
                shutil.rmtree(r)
                print(f"🧹 Deleted old run directory: {r}")
            except Exception as e:
                print(f"⚠️ Failed to delete {r}: {e}")


    print(f"\n🎉 Training finished. Results saved in: {current_run_dir}")
    return True

# def run_yolo_inference(
#     model_name: str,
#     device: str = "0",
#     conf: float = 0.2,
#     **kwargs
# ) -> bool:
#     """
#     Enhanced YOLOv8 inference pipeline with proper class tracking.
#     """
    
#     def compute_iou(box1, box2):
#         """Calculate IoU between two boxes [x1, y1, x2, y2]."""
#         x1 = max(box1[0], box2[0])
#         y1 = max(box1[1], box2[1])
#         x2 = min(box1[2], box2[2])
#         y2 = min(box1[3], box2[3])
        
#         intersection = max(0, x2 - x1) * max(0, y2 - y1)
#         area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
#         area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
#         union = area1 + area2 - intersection
        
#         return intersection / union if union > 0 else 0
    
#     print(f"\n--- YOLOv8 Inference Pipeline ({model_name}) ---")

#     model_path = MODELS_ROOT / model_name / "best.pt"
#     if not model_path.exists():
#         print(f"❌ Model not found: {model_path}")
#         print(f"💡 Did you train the model? Run training first with 14 keypoints.")
#         return False

#     print(f"📦 Loading model from: {model_path}")
#     model = YOLO(str(model_path))
    
#     # Verify model has correct number of keypoints
#     if hasattr(model.model, 'kpt_shape'):
#         kpt_shape = model.model.kpt_shape
#         print(f"🔍 Model keypoint shape: {kpt_shape}")
#         if kpt_shape[0] != 14:
#             print(f"❌ ERROR: Model expects {kpt_shape[0]} keypoints, but we need 14!")
#             print(f"💡 You need to train a custom model with 14 keypoints first.")
#             return False
#     device = "cuda" if torch.cuda.is_available() else device
#     print(f"Using device: {device}")

#     image_paths = sorted([
#         p for p in IMAGES_TEST.glob("*")
#         if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
#     ])
#     if not image_paths:
#         print(f"❌ No test images found at: {IMAGES_TEST}")
#         return False

#     print(f"Found {len(image_paths)} test images")

#     all_results: list[Results] = []

#     for img_id, img_path in enumerate(image_paths, start=1):
#         image = Image.open(img_path).convert("RGB")
#         W, H = image.width, image.height

#         # Load RF-DETR annotations
#         annotation_path = RFDETR_ANNOTS_DIR / f"{img_path.stem}.json"
#         detections = []
#         if annotation_path.exists():
#             with open(annotation_path, "r", encoding="utf-8") as f:
#                 data = json.load(f)
#                 detections = data.get("detections", [])

#         # --- CASE 1: Use crops if detections exist ---
#         if detections:
#             print(f"→ Processing {img_path.name}: {len(detections)} RF-DETR detections")
#             combined_kps = []
#             combined_boxes = []
#             combined_classes = []  # Track class names

#             for det_idx, det in enumerate(detections):
#                 bbox_xyxy = det["bbox_xyxy"]
#                 class_name = det.get("class_name", "boxer_blue")  # Get class from RF-DETR
                
#                 x1, y1, x2, y2 = map(int, bbox_xyxy)
#                 x1, y1 = max(0, x1), max(0, y1)
#                 x2, y2 = min(W, x2), min(H, y2)
                
#                 crop_w, crop_h = x2 - x1, y2 - y1
                
#                 print(f"  🔍 Detection {det_idx} ({class_name}): bbox=({x1},{y1},{x2},{y2}), size={crop_w}x{crop_h}")
                
#                 # More lenient crop size check
#                 if crop_w <= 0 or crop_h <= 0 or crop_w < 30 or crop_h < 30:
#                     print(f"  ⚠️ Skipping too-small crop: {crop_w}x{crop_h}")
#                     continue

#                 # Add padding to crops for better detection
#                 pad = 20
#                 x1_pad = max(0, x1 - pad)
#                 y1_pad = max(0, y1 - pad)
#                 x2_pad = min(W, x2 + pad)
#                 y2_pad = min(H, y2 + pad)
                
#                 cropped = image.crop((x1_pad, y1_pad, x2_pad, y2_pad))
#                 pad_offset_x = x1_pad
#                 pad_offset_y = y1_pad

#                 # Lower confidence threshold for crops
#                 results = model.predict(
#                     cropped,
#                     imgsz=640,
#                     device=device,
#                     conf=0.2,  # Lower threshold
#                     verbose=False
#                 )

#                 if not results or len(results) == 0:
#                     print(f"  ⚠️ No YOLO results for detection {det_idx}")
#                     continue

#                 res = results[0]
#                 if not hasattr(res, "keypoints") or res.keypoints is None or len(res.keypoints) == 0:
#                     print(f"  ⚠️ No keypoints for detection {det_idx} (model returned {len(res.boxes) if hasattr(res, 'boxes') else 0} boxes)")
#                     continue

#                 # Get ALL detections (not just best one)
#                 kp_data = res.keypoints.data.cpu().numpy()
#                 boxes = res.boxes.xyxy.cpu().numpy()
#                 confs = res.boxes.conf.cpu().numpy()
                
#                 print(f"  📊 YOLO found {len(kp_data)} detections with confs: {confs}")
                
#                 if len(kp_data) == 0:
#                     continue
                
#                 # Keep ALL detections, but filter by confidence
#                 valid_mask = confs > 0.2
#                 kp_data = kp_data[valid_mask]
#                 boxes = boxes[valid_mask]
                
#                 if len(kp_data) == 0:
#                     print(f"  ⚠️ No detections above confidence threshold")
#                     continue
                
#                 # Shift keypoints to original frame coordinates (accounting for padding)
#                 kp_data[..., 0] += pad_offset_x  # x coordinates
#                 kp_data[..., 1] += pad_offset_y  # y coordinates
                
#                 # Shift boxes
#                 boxes[:, [0, 2]] += pad_offset_x
#                 boxes[:, [1, 3]] += pad_offset_y

#                 combined_kps.append(kp_data)
#                 combined_boxes.append(boxes)
#                 # Store class name for EACH detection in this crop
#                 combined_classes.extend([class_name] * len(kp_data))
                
#                 print(f"  ✅ Detection {det_idx} ({class_name}): {len(kp_data)} poses found")

#             if combined_kps:
#                 # Merge all detections from this image
#                 kps_final = np.concatenate(combined_kps, axis=0)
#                 boxes_final = np.concatenate(combined_boxes, axis=0)

#                 # Convert to torch
#                 boxes_crops = torch.tensor(boxes_final, dtype=torch.float32)
#                 conf_t = torch.ones((boxes_crops.shape[0], 1)) * 0.99
#                 cls_t = torch.zeros((boxes_crops.shape[0], 1))
#                 boxes_yolo = torch.cat([boxes_crops, conf_t, cls_t], dim=1)

#                 # Construct synthetic YOLO Results object
#                 result_obj = Results(
#                     orig_img=np.array(image),
#                     path=str(img_path),
#                     names=model.names,
#                     boxes=boxes_yolo,
#                     keypoints=torch.tensor(kps_final, dtype=torch.float32),
#                 )
                
#                 # **CRITICAL**: Attach class names to Results object
#                 result_obj.class_names = combined_classes
                
#                 all_results.append(result_obj)
#                 print(f"  ✅ Total poses for {img_path.name}: {len(kps_final)}")
#             else:
#                 # Fallback: If crops failed, try full-frame with RF-DETR class info
#                 print(f"  ⚠️ No poses from crops, trying full-frame for {img_path.name}")
#                 results = model.predict(
#                     source=str(img_path),
#                     imgsz=640,
#                     device=device,
#                     conf=conf,
#                     verbose=False
#                 )
#                 if results and len(results[0].keypoints) > 0:
#                     # Try to match detections to RF-DETR classes by IoU
#                     results[0].class_names = []
#                     yolo_boxes = results[0].boxes.xyxy.cpu().numpy()
                    
#                     for yolo_box in yolo_boxes:
#                         best_match = "boxer_blue"
#                         best_iou = 0
#                         for det in detections:
#                             rf_box = det["bbox_xyxy"]
#                             iou = compute_iou(yolo_box, rf_box)
#                             if iou > best_iou:
#                                 best_iou = iou
#                                 best_match = det.get("class_name", "boxer_blue")
#                         results[0].class_names.append(best_match)
                    
#                     all_results.append(results[0])
#                     print(f"  ✅ Full-frame found {len(results[0].keypoints)} poses")
#                 else:
#                     print(f"  ⚠️ No valid keypoints detected for {img_path.name}")

#         # --- CASE 2: Fallback to full-frame inference ---
#         else:
#             print(f"→ No RF-DETR detections, using full-frame inference for {img_path.name}")
#             results = model.predict(
#                 source=str(img_path),
#                 imgsz=640,
#                 device=device,
#                 conf=conf,
#                 verbose=False
#             )
#             if results:
#                 # Without RF-DETR, we can't determine classes
#                 results[0].class_names = []
#                 all_results.append(results[0])

#         if img_id % 10 == 0:
#             print(f"Progress: {img_id}/{len(image_paths)} images...")

#     print(f"\n✅ Total results collected: {len(all_results)}")
#     total_annotations = sum(len(r.keypoints) if hasattr(r, 'keypoints') and r.keypoints is not None else 0 
#                            for r in all_results)
#     print(f"✅ Total pose annotations: {total_annotations}")

#     # Save to COCO format
#     print(f"\nSaving results to COCO format at {OUT_TEST_JSON} ...")
#     save_results_as_coco(
#         results=all_results,
#         image_paths=image_paths,
#         out_json=OUT_TEST_JSON,
#         overwrite=True
#     )

#     print(f"✅ Inference complete. COCO annotations saved to: {OUT_TEST_JSON}")
#     return True

# =================================================================================================================================================

# def run_yolo_inference(
#     model_name: str, 
#     device: str = "0",
#     conf: float = 0.2,
#     **kwargs # Accept other args
# ) -> bool:
#     """
#     Runs YOLO inference on test images and saves results as COCO JSON.
#     (Content is copied directly from original inference.py:run_inference)
#     """

#     print(f"\n--- YOLOv8 Inference Pipeline ({model_name}) ---")
#     model_path = MODELS_ROOT / model_name / "best.pt"
#     if not model_path.exists():
#         print(f"❌ Model not found: {model_path}")
#         return False

#     print(f"Loading YOLO model from: {model_path}")
#     model = YOLO(str(model_path))

#     # Collect test images (sorted for consistency)
#     image_paths = sorted([
#         p for p in IMAGES_TEST.glob("*") 
#         if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
#     ])
    
#     if not image_paths:
#         print(f"No test images found at: {IMAGES_TEST}")
#         return True # Not an error if no test images

#     print(f"Found {len(image_paths)} test images")
#     print(f"Running inference with confidence threshold: {conf}")
    
#     # Run inference
#     results = model.predict(
#         source=[str(p) for p in image_paths],
#         imgsz=640,
#         save=False,
#         device=device,
#         conf=conf,
#         verbose=True
#     )

#     results = list(results)
#     print(f"\nConverting YOLO results to COCO format...")
    
#     # Convert results to COCO format and save
#     save_results_as_coco(
#         results=results,
#         image_paths=image_paths,
#         out_json=OUT_TEST_JSON,
#         overwrite=True
#     )

#     print(f"✅ YOLO Inference complete! COCO annotations saved to: {OUT_TEST_JSON}")
#     return True

# =========================================================================================================================================

def get_next_output_number(output_dir):
    """Find the next available output number"""
    existing_files = list(output_dir.glob("output_*.mp4"))
    if not existing_files:
        return 1
    
    numbers = []
    for file in existing_files:
        try:
            num = int(file.stem.split('_')[1])
            numbers.append(num)
        except (IndexError, ValueError):
            continue
    
    return max(numbers) + 1 if numbers else 1

def select_video_file():
    """Open file dialog to select video"""
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    file_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
            ("All files", "*.*")
        ]
    )
    root.destroy()
    return file_path

def run_yolo_inference(
    model_name: str, 
    device: str = "0",
    conf: float = 0.7,
    target_fps: int = 25,
    **kwargs # Accept other args
) -> bool:
    """
    Runs YOLO inference on a selected video and saves annotated output with connected keypoints.
    
    Args:
        model_name: Name of the model folder
        device: GPU device (e.g., "0" for cuda:0)
        conf: Confidence threshold
        target_fps: Target FPS for processing and output (default: 25)
    """    
    # Ensure conf is Python float, not numpy float
    conf = float(conf)

    print(f"\n--- YOLOv8 Inference Pipeline ({model_name}) ---")
    
    # Model path (unchanged)
    model_path = MODELS_ROOT / model_name / "best.pt"
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return False

    print(f"Loading YOLO model from: {model_path}")
    model = YOLO(str(model_path))

    # Select video using tkinter
    print("\nPlease select a video file...")
    video_path = select_video_file()
    
    if not video_path:
        print("No video file selected. Exiting...")
        return False
    
    print(f"Selected video: {video_path}")
    
    # Setup output directory (unchanged)
    output_dir = Path(__file__).resolve().parents[2] / "data" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get next output number
    output_num = get_next_output_number(output_dir)
    output_path = output_dir / f"output_{output_num}.mp4"
    
    # Open video to get properties
    cap = cv2.VideoCapture(video_path)
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    print(f"Original video: {width}x{height} @ {original_fps} FPS, {total_frames} frames")
    print(f"Target FPS: {target_fps}")
    print(f"Output will be saved as: {output_path}")
    print(f"Running inference with confidence threshold: {conf}")
    print("Starting pose estimation...\n")
    
    # Calculate frame skip interval
    frame_skip = max(1, round(original_fps / target_fps))
    
    # Open video for reading
    cap = cv2.VideoCapture(video_path)
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, target_fps, (width, height))
    
    frame_idx = 0
    processed_frames = 0
    
    # Custom 14 keypoint skeleton connections
    # Keypoints: left_shoulder(0), right_shoulder(1), left_hip(2), right_hip(3),
    #            left_knee(4), right_knee(5), left_ankle(6), right_ankle(7),
    #            left_elbow(8), left_wrist(9), right_elbow(10), right_wrist(11),
    #            neck(12), nose(13)
    # Convert from 1-indexed to 0-indexed
    skeleton = [
        (8, 9),   # left_elbow to left_wrist
        (0, 2),   # left_shoulder to left_hip
        (1, 3),   # right_shoulder to right_hip
        (0, 1),   # left_shoulder to right_shoulder
        (2, 3),   # left_hip to right_hip
        (5, 7),   # right_knee to right_ankle
        (3, 5),   # right_hip to right_knee
        (4, 6),   # left_knee to left_ankle
        (10, 11), # right_elbow to right_wrist
        (13, 12), # nose to neck
        (0, 12),  # left_shoulder to neck
        (12, 1),  # neck to right_shoulder
        (1, 10),  # right_shoulder to right_elbow
        (0, 8),   # left_shoulder to left_elbow
        (2, 4)    # left_hip to left_knee
    ]
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames to match target FPS
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue
            
            # Run inference on single frame
            results = model.predict(
                source=frame,
                imgsz=640,
                save=False,
                device=device,
                conf=conf,
                verbose=False
            )
            
            # Start with original frame
            annotated_frame = frame.copy()
            
            # Draw bounding boxes and keypoints
            if results[0].keypoints is not None and len(results[0].keypoints) > 0:
                keypoints = results[0].keypoints.xy.cpu().numpy()  # Shape: (num_persons, num_keypoints, 2)
                
                # Draw for each detected person
                for person_kpts in keypoints:
                    # Draw skeleton connections
                    for start_idx, end_idx in skeleton:
                        if start_idx < len(person_kpts) and end_idx < len(person_kpts):
                            start_point = person_kpts[start_idx]
                            end_point = person_kpts[end_idx]
                            
                            # Only draw if both keypoints are detected (not at 0,0)
                            if (start_point[0] > 0 and start_point[1] > 0 and 
                                end_point[0] > 0 and end_point[1] > 0):
                                cv2.line(
                                    annotated_frame,
                                    (int(start_point[0]), int(start_point[1])),
                                    (int(end_point[0]), int(end_point[1])),
                                    (0, 255, 0),  # Green color
                                    2,            # Line thickness
                                    cv2.LINE_AA
                                )
                    
                    # Draw keypoints on top of lines
                    for kpt in person_kpts:
                        if kpt[0] > 0 and kpt[1] > 0:  # Only draw if keypoint is detected
                            cv2.circle(
                                annotated_frame,
                                (int(kpt[0]), int(kpt[1])),
                                5,              # Radius
                                (0, 0, 255),    # Red color
                                -1,             # Filled circle
                                cv2.LINE_AA
                            )
            
            # Draw bounding boxes if available
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                
                for box, conf_val in zip(boxes, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    # Convert numpy float to Python float
                    conf_float = float(conf_val)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(
                        annotated_frame,
                        f'{conf_float:.2f}',
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2
                    )
            
            # Write frame
            out.write(annotated_frame)
            
            processed_frames += 1
            frame_idx += 1
            
            # Print progress every 50 frames
            if processed_frames % 50 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"Processing: {progress:.1f}% (Frame {frame_idx}/{total_frames}, Output frames: {processed_frames})")
    
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        return False
    
    finally:
        cap.release()
        out.release()
    
    print(f"\n✅ YOLO Inference complete!")
    print(f"   Processed {processed_frames} frames from {total_frames} total frames")
    print(f"   Video saved to: {output_path}")
    return True