"""
src/pipelines/yolo_pipeline.py (Memory Optimized with Symlinks)

Enhanced YOLO Pose training script:
- Creates symlinks in /data/processed/images/{split} pointing to original images
- Creates labels in /data/processed/labels/{split}
- This satisfies YOLO's expected directory structure without duplicating images
- Saves model weights to /models/{model_name}/ (best.pt, last.pt)
- Saves training results (graphs, metrics, etc.) to /runs/{model_name}/run_{n}/
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
import json
import numpy as np

# Import your converter
from ..utils.coco_to_yolo_pose import coco_to_yolo_keypoints
from ..utils.results_to_coco import save_results_as_coco

# -----------------------
# Project paths
# -----------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
ANNOTS_DIR = DATA_PROCESSED / "annotations"

# Virtual directories for YOLO (symlinks + labels)
IMAGES_DIR = DATA_PROCESSED / "images"  # Symlinks to original images
LABELS_DIR = DATA_PROCESSED / "labels"  # Real label files

MODELS_ROOT = PROJECT_ROOT / "models"
OUT_TEST_JSON = ANNOTS_DIR / "test.json"
RUNS_ROOT = PROJECT_ROOT / "runs"

# -----------------------
# Defaults
# -----------------------
DEFAULT_MODEL = "yolov8m-pose"
NUM_KEYPOINTS = 14
EPOCHS = 50
IMGSZ = 640
DEFAULT_BATCH = 16
DEFAULT_WORKERS = 0

WEIGHTS_MAP = {
    "yolov8s-pose": "yolov8s-pose.pt",
    "yolov8m-pose": "yolov8m-pose.pt",
    "yolov11s-pose": "yolo11s-pose.pt",
    "yolov11m-pose": "yolo11m-pose.pt",
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


def prepare_virtual_dataset(split: str, json_path: Path):
    """
    Creates virtual dataset structure for YOLO:
    1. Reads the JSON for the split (train/val)
    2. Creates directories: data/processed/images/{split} & labels/{split}
    3. Creates symlinks in images/{split} pointing to original image locations
    4. Generates YOLO .txt labels in labels/{split}
    
    This structure allows YOLO to find both images and labels without duplicating image files.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {json_path}")

    # Define target directories
    img_split_dir = IMAGES_DIR / split
    lbl_split_dir = LABELS_DIR / split
    
    # Cleanup and recreate
    if img_split_dir.exists():
        shutil.rmtree(img_split_dir)
    if lbl_split_dir.exists():
        shutil.rmtree(lbl_split_dir)
    img_split_dir.mkdir(parents=True, exist_ok=True)
    lbl_split_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🏗️  Preparing Virtual Dataset for '{split}'...")
    print(f"   Images (Symlinks): {img_split_dir}")
    print(f"   Labels (Real):     {lbl_split_dir}")

    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 1. Create Symlinks to original images
    symlinks_created = 0
    symlinks_failed = 0
    
    for img in data.get('images', []):
        original_path = Path(img['file_name'])
        
        if not original_path.exists():
            print(f"⚠️ Warning: Original image missing: {original_path}")
            continue
            
        # Create symlink path: processed/images/train/image123.jpg
        symlink_path = img_split_dir / original_path.name
        
        try:
            # Remove if exists (cleanup)
            if symlink_path.exists() or symlink_path.is_symlink():
                symlink_path.unlink()
            
            # Create symlink
            os.symlink(original_path, symlink_path)
            symlinks_created += 1
            
        except OSError as e:
            print(f"❌ Failed to create symlink for {original_path.name}: {e}")
            print(f"   Note: On Windows, you may need to run as Administrator or enable Developer Mode")
            symlinks_failed += 1
            
            if symlinks_failed >= 5:
                print("\n❌ Too many symlink failures. Please:")
                print("   1. Run as Administrator, OR")
                print("   2. Enable Windows Developer Mode:")
                print("      Settings > Update & Security > For Developers > Developer Mode: ON")
                raise RuntimeError("Cannot create symlinks. See suggestions above.")

    if symlinks_created == 0:
        raise RuntimeError(f"No symlinks created for {split} split. Check image paths in {json_path}")
    
    print(f"✅ Created {symlinks_created} symlinks")

    # 2. Generate YOLO Labels
    # Pass the symlink directory as images_dir since that's where YOLO will look
    print(f"📝 Converting annotations to YOLO format...")
    coco_to_yolo_keypoints(
        str(json_path),
        str(img_split_dir),  # Directory where images 'appear' to be (symlinks)
        str(lbl_split_dir),  # Directory where labels should be saved
        num_keypoints=NUM_KEYPOINTS,
        clear_out=False  # We already cleared it
    )

    print(f"✅ Virtual dataset prepared for '{split}'")
    return img_split_dir


def dataset_yaml_path_for(model_name: str) -> Path:
    """
    Create dataset YAML for YOLO training.
    Points to the virtual dataset structure with symlinks.
    """
    train_json = ANNOTS_DIR / "train.json"
    val_json = ANNOTS_DIR / "val.json"
    
    # Prepare virtual datasets (Symlinks + Labels)
    train_img_dir = prepare_virtual_dataset("train", train_json)
    val_img_dir = prepare_virtual_dataset("val", val_json)
    
    yaml_path = PROJECT_ROOT / f"{model_name}_dataset.yaml"
    content = {
        "path": str(DATA_PROCESSED),  # Base path
        "train": "images/train",      # Relative to path
        "val": "images/val",          # Relative to path
        "kpt_shape": [NUM_KEYPOINTS, 3],
        "names": ["boxer_red", "boxer_blue"],
        "keypoints": KEYPOINTS,
        "skeleton": SKELETON,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(content, f)
    
    print(f"📝 Created dataset YAML: {yaml_path}")
    print(f"   Path: {DATA_PROCESSED}")
    print(f"   Train: images/train (with {len(list(train_img_dir.glob('*')))} symlinks)")
    print(f"   Val: images/val (with {len(list(val_img_dir.glob('*')))} symlinks)")
    
    return yaml_path


def labels_exist(split: str) -> bool:
    """Check if YOLO labels exist for a split."""
    dirp = LABELS_DIR / split
    return dirp.exists() and any(dirp.glob("*.txt"))


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

    # Always recreate virtual dataset (symlinks + labels)
    # This ensures consistency if source images or annotations changed
    print("\n" + "=" * 60)
    print("Preparing Virtual Dataset Structure")
    print("=" * 60)
    
    data_yaml = dataset_yaml_path_for(model_name)
    
    print("\n✅ Virtual dataset structure ready")
    print("   - Symlinks point to original image locations")
    print("   - Labels generated in standard YOLO format")
    print("   - No image duplication!")

    # Prepare weights & checkpoint
    local_weights = ensure_pretrained(model_name)
    checkpoint_target = model_dir / "best.pt"

    if resume_from_best and checkpoint_target.exists():
        print(f"\n🔄 Resuming from checkpoint: {checkpoint_target}")
        weights_to_load = str(checkpoint_target)
    else:
        weights_to_load = str(local_weights)
        print(f"\n📦 Using pretrained weights: {weights_to_load}")

    # Create YOLO model
    os.chdir(PROJECT_ROOT)
    model = YOLO(weights_to_load)

    # Create new run directory
    run_number = 1
    while (RUNS_ROOT / model_name / f"run_{run_number}").exists():
        run_number += 1
    current_run_dir = RUNS_ROOT / model_name / f"run_{run_number}"
    current_run_dir.mkdir(parents=True)
    print(f"🗂️ Logging this training to: {current_run_dir}")

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
                project=str(current_run_dir),
                name="",  # avoid subfolder (no /exp)
                exist_ok=True,
                device=0 if torch.cuda.is_available() else "cpu"
            )
            
            # Print validation summary
            if isinstance(results, dict):
                print("\n📊 Validation summary:")
                print(results)
            
            break  # Success!
            
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

    # Copy best and last weights
    exp_weights_dir = current_run_dir / "train" / "weights"
    best_src = exp_weights_dir / "best.pt"
    last_src = exp_weights_dir / "last.pt"

    if best_src.exists():
        shutil.copy2(best_src, model_dir / "best.pt")
        print(f"\n✅ Copied best.pt -> {model_dir / 'best.pt'}")
    else:
        print("\n⚠️ best.pt not found in run folder.")

    if last_src.exists():
        shutil.copy2(last_src, model_dir / "last.pt")
        print(f"✅ Copied last.pt -> {model_dir / 'last.pt'}")
    else:
        print("⚠️ last.pt not found in run folder.")

    # Keep only last 5 runs
    model_runs_dir = RUNS_ROOT / model_name
    if model_runs_dir.exists():
        runs = sorted(
            [p for p in model_runs_dir.glob("run_*") if p.is_dir()],
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


def get_test_images_from_annotations() -> list:
    """
    Get test image paths from test.json annotations.
    Returns list of Path objects.
    """
    if not OUT_TEST_JSON.exists():
        print(f"⚠️ Test annotations not found at: {OUT_TEST_JSON}")
        return []
    
    with open(OUT_TEST_JSON, 'r') as f:
        data = json.load(f)
    
    images = data.get('images', [])
    if not images:
        print("⚠️ No images found in test annotations")
        return []
    
    # Extract paths from annotations
    image_paths = []
    for img in images:
        img_path = Path(img['file_name'])
        if img_path.exists():
            image_paths.append(img_path)
        else:
            print(f"⚠️ Image not found: {img_path}")
    
    return sorted(image_paths)


def run_yolo_inference(
    model_name: str, 
    device: str = "0",
    conf: float = 0.2,
    **kwargs  # Accept other args
) -> bool:
    """
    Runs YOLO inference on test images and saves results as COCO JSON.
    Reads test image paths from annotations.
    """
    print(f"\n--- YOLOv8 Inference Pipeline ({model_name}) ---")
    model_path = MODELS_ROOT / model_name / "best.pt"
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return False

    print(f"Loading YOLO model from: {model_path}")
    model = YOLO(str(model_path))

    # Get test images from annotations
    image_paths = get_test_images_from_annotations()
    
    if not image_paths:
        print(f"⚠️ No test images found")
        return True  # Not an error if no test images

    print(f"Found {len(image_paths)} test images")
    print(f"Running inference with confidence threshold: {conf}")
    
    # Run inference
    results = model.predict(
        source=[str(p) for p in image_paths],
        imgsz=640,
        save=False,
        device=device,
        conf=conf,
        verbose=True,
        half=True
    )

    # Debug first result
    results_list = list(results)
    if results_list:
        for i, result in enumerate(results_list[:1]):
            print(f"\n=== Debug Result {i} ===")
            print(f"Has boxes: {hasattr(result, 'boxes')}")
            if hasattr(result, 'boxes') and result.boxes is not None:
                print(f"Boxes shape: {result.boxes.xyxy.shape}")
                print(f"Has cls: {hasattr(result.boxes, 'cls')}")
                if hasattr(result.boxes, 'cls'):
                    print(f"Classes: {result.boxes.cls}")
                    print(f"Class names: {result.names}")

    print(f"\nConverting YOLO results to COCO format...")
    
    # Convert results to COCO format and save
    save_results_as_coco(
        results=results_list,
        image_paths=image_paths,
        out_json=OUT_TEST_JSON,
        overwrite=True
    )

    print(f"✅ YOLO Inference complete! COCO annotations saved to: {OUT_TEST_JSON}")
    return True