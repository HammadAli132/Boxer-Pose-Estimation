"""
src/training/train.py

DISPATCHER: Reads --model arg and calls the appropriate pipeline.
"""
import sys
import argparse
from pathlib import Path
import os

from ..pipelines.yolo_pipeline import run_yolo_training
# from ..pipelines.dino_pipeline import run_dino_training

# -----------------------
# Defaults
# -----------------------
DEFAULT_MODEL = "yolov8s-pose"
EPOCHS = 50
IMGSZ = 640
DEFAULT_BATCH = 8
DEFAULT_WORKERS = 0

# --- DISPATCHER LOGIC ---
def main():
    parser = argparse.ArgumentParser(description="Train Pose Models")
    # NOTE: The 'model' arg is now required and is passed from main.py
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name (yolov8s-pose or dinov2_vits14)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    
    # Convert args to a dictionary for easy passing
    kwargs = vars(args)

    success = False
    if args.model.startswith("yolov"):
        # The existing YOLO code is in yolo_pipeline.py
        success = run_yolo_training(**kwargs)
        
    # elif args.model.startswith("dinov2"):
    #     # The new DinoV2 code will be in dino_pipeline.py
    #     success = run_dino_training(**kwargs)
        
    else:
        print(f"❌ Unknown model: {args.model}")
        sys.exit(1)

    if not success:
        sys.exit(2)

if __name__ == "__main__":
    main()