"""
src/training/train.py

Robust training entry point for YOLO pose models.
- Validates labels (YOLO format) and auto-converts from COCO if needed.
- Downloads pretrained weights into /models/{model_name}/.
- Trains with configurable batch/workers, retries with smaller config on memory errors.
- Saves best.pt to /models/{model_name}/best.pt for future resume.
- Exposes main() for your central launcher.
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

# Import your converter (assumes package path resolves; adjust if needed)
from utils.coco_to_yolo_pose import coco_to_yolo_keypoints

# -----------------------
# Project paths (relative)
# -----------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]  # go up from src/training -> project root
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
IMAGES_DIR = DATA_PROCESSED / "images"
ANNOTS_DIR = DATA_PROCESSED / "annotations"
LABELS_DIR = DATA_PROCESSED / "labels"     # will contain labels/train/*.txt, labels/val/*.txt
MODELS_ROOT = PROJECT_ROOT / "models"

# -----------------------
# Default training config
# -----------------------
DEFAULT_MODEL = "yolov8s-pose"   # change if you want a different default
NUM_KEYPOINTS = 14
EPOCHS = 50
IMGSZ = 640
DEFAULT_BATCH = 8   # reasonable default for 8GB GPU; lowered vs 16
DEFAULT_WORKERS = 0 # safer on Windows; change to 4+ on Linux

# Map model name -> pretrained weight filename in assets repo
WEIGHTS_MAP = {
    "yolov8s-pose": "yolov8s-pose.pt",
    "yolov11-pose": "yolov11-pose.pt",
}

# Custom keypoints & skeleton (as you provided)
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
    """Ensure pretrained weights are present in /models/{model_name}/ and return local path."""
    if model_name not in WEIGHTS_MAP:
        raise ValueError(f"Unknown model_name: {model_name}")

    model_dir = MODELS_ROOT / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    local_file = model_dir / WEIGHTS_MAP[model_name]

    if local_file.exists():
        return local_file

    # Download from Ultralytics assets release
    url = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{WEIGHTS_MAP[model_name]}"
    print(f"⬇️ Downloading pretrained weights for {model_name} to {local_file} ...")
    download_file(url, local_file)
    print("✅ Download complete.")
    return local_file


def dataset_yaml_path_for(model_name: str) -> Path:
    """Create dataset yaml (in project root) that points to processed data and keypoint info."""
    yaml_path = PROJECT_ROOT / f"{model_name}_dataset.yaml"
    content = {
        "path": str(DATA_PROCESSED),
        "train": str(IMAGES_DIR / "train"),
        "val": str(IMAGES_DIR / "val"),
        "kpt_shape": [NUM_KEYPOINTS, 3],
        "names": ["quadruped", "person"],
        "keypoints": KEYPOINTS,
        "skeleton": SKELETON,
    }
    import yaml as _yaml
    with open(yaml_path, "w") as f:
        _yaml.dump(content, f)
    return yaml_path


def labels_exist(split: str) -> bool:
    """Return True if YOLO .txt labels exist for the given split (train/val)."""
    dirp = LABELS_DIR / split
    if not dirp.exists():
        return False
    txts = list(dirp.glob("*.txt"))
    return len(txts) > 0


def ensure_labels_from_coco(split: str, coco_json: Path, images_dir: Path, out_labels_dir: Path, num_kpts: int):
    """
    Run COCO->YOLO converter to produce per-image .txt labels in out_labels_dir.
    This will overwrite/append existing labels (converter uses append in your version),
    so we remove existing label dir first for cleanliness.
    """
    if out_labels_dir.exists():
        # clear it to avoid duplicates
        shutil.rmtree(out_labels_dir)
    out_labels_dir.mkdir(parents=True, exist_ok=True)
    print(f"🔁 Converting {coco_json} -> {out_labels_dir} ...")
    coco_to_yolo_keypoints(str(coco_json), str(images_dir), str(out_labels_dir), num_keypoints=num_kpts)
    print("✅ Conversion done.")


# -----------------------
# Core training logic
# -----------------------
def run_training(
    model_name: str = DEFAULT_MODEL,
    epochs: int = EPOCHS,
    imgsz: int = IMGSZ,
    batch: int = DEFAULT_BATCH,
    workers: int = DEFAULT_WORKERS,
    force_convert_labels: bool = False,
    resume_from_best: bool = True,
):
    # Paths
    model_dir = MODELS_ROOT / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # JSON annotation paths
    train_json = ANNOTS_DIR / "train.json"
    val_json = ANNOTS_DIR / "val.json"

    if not train_json.exists() or not val_json.exists():
        raise FileNotFoundError(f"Train/Val JSONs not found under {ANNOTS_DIR}")

    # ensure YOLO labels exist, else convert (unless user turned off)
    train_labels_dir = LABELS_DIR / "train"
    val_labels_dir = LABELS_DIR / "val"
    if force_convert_labels or not (labels_exist("train") and labels_exist("val")):
        ensure_labels_from_coco("train", train_json, IMAGES_DIR / "train", train_labels_dir, NUM_KEYPOINTS)
        ensure_labels_from_coco("val", val_json, IMAGES_DIR / "val", val_labels_dir, NUM_KEYPOINTS)

    # quick re-check:
    if not (labels_exist("train") and labels_exist("val")):
        print("❌ ERROR: labels still missing after attempted conversion.")
        print(f"Check {train_labels_dir} and {val_labels_dir} for .txt files.")
        return False

    # ensure pretrained weights available locally and get path
    local_weights = ensure_pretrained(model_name)

    # dataset yaml
    data_yaml = dataset_yaml_path_for(model_name)

    # optional resume checkpoint
    checkpoint_target = model_dir / "best.pt"
    if resume_from_best and checkpoint_target.exists():
        print(f"🔁 Resuming from existing checkpoint: {checkpoint_target}")
        weights_to_load = str(checkpoint_target)
    else:
        weights_to_load = str(local_weights)
        print(f"📦 Using pretrained weights: {weights_to_load}")

    # force CWD to project root so ultralytics writes runs into project path
    os.chdir(PROJECT_ROOT)

    # create YOLO model object
    model = YOLO(weights_to_load)

    # training with retry on memory/paging errors
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
                project=str(model_dir),   # ensure saved under /models/{model_name}/exp
                name="exp",
                exist_ok=True,
                device=0 if torch.cuda.is_available() else "cpu"
            )
            # success
            break

        except Exception as e:
            # detect likely paging / memory error (WinError 1455 or CUDA OOM)
            tb = traceback.format_exc()
            print(f"\n⚠️ Training failed on attempt {attempt} with error:\n{e}\n")
            if "WinError 1455" in tb or "out of memory" in tb.lower() or "paging file" in tb.lower():
                # reduce batch/workers and retry
                if current_batch > 1:
                    current_batch = max(1, current_batch // 2)
                else:
                    current_batch = 1
                if current_workers > 0:
                    current_workers = max(0, current_workers - 1)
                print(f"🔧 Retrying with batch={current_batch}, workers={current_workers} ...")
                time.sleep(2)
                continue
            else:
                # non-memory error — abort
                print("❌ Non-memory error encountered. Aborting training.")
                print(tb)
                raise

    else:
        print("❌ All training attempts failed.")
        return False

    # copy best checkpoint (if produced)
    exp_weights = model_dir / "exp" / "weights" / "best.pt"
    if exp_weights.exists():
        shutil.copy2(exp_weights, model_dir / "best.pt")
        print(f"\n✅ Copied best.pt -> {model_dir / 'best.pt'}")
    else:
        print("\n⚠️ best.pt not found in run directory. Check run folder.")

    # try a validation run (prints metrics)
    try:
        val_results = model.val(data=str(data_yaml))
        print("\n📊 Validation summary (ultralytics results):")
        # try to print some useful keys
        if hasattr(val_results, "results_dict"):
            print(val_results.results_dict)
        else:
            print(val_results)
    except Exception as e:
        print("⚠️ Validation failed:", e)

    return True


# -----------------------
# CLI main() for menu caller
# -----------------------
def main():
    parser = argparse.ArgumentParser(prog="train.py", description="Train YOLO pose models")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="model name (yolov8s-pose or yolov11-pose)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--force-convert", action="store_true", help="force COCO->YOLO conversion even if labels exist")
    parser.add_argument("--no-resume", action="store_true", help="don't resume from best.pt even if present")
    args = parser.parse_args()

    success = run_training(
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        force_convert_labels=args.force_convert,
        resume_from_best=not args.no_resume
    )
    if not success:
        sys.exit(2)


if __name__ == "__main__":
    main()
