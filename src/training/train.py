"""
src/training/train.py

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

# Import your converter
from utils.coco_to_yolo_pose import coco_to_yolo_keypoints

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
RUNS_ROOT = PROJECT_ROOT / "runs"  # <-- NEW: store training results here

# -----------------------
# Defaults
# -----------------------
DEFAULT_MODEL = "yolov8s-pose"
NUM_KEYPOINTS = 14
EPOCHS = 50
IMGSZ = 640
DEFAULT_BATCH = 8
DEFAULT_WORKERS = 0

WEIGHTS_MAP = {
    "yolov8s-pose": "yolov8s-pose.pt",
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
        "names": ["quadruped", "person"],
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
def run_training(
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
    while (RUNS_ROOT / f"run_{run_number}").exists():
        run_number += 1
    current_run_dir = RUNS_ROOT / f"run_{run_number}"
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
                project=str(current_run_dir),  # now goes to /runs/run_{n}
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


# -----------------------
# CLI Entry
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Train YOLO Pose Models")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    success = run_training(
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        resume_from_best=not args.no_resume,
    )

    if not success:
        sys.exit(2)


if __name__ == "__main__":
    main()