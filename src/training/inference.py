import argparse
from pathlib import Path
import sys
import torch
import subprocess
import os

# Add src to path for imports
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# --- DEFAULT CONSTANTS ---
DEFAULT_MODEL_NAME = "yolov8m-pose"
DEFAULT_CONF_THRESHOLD = 0.2

# --- MAIN DISPATCHER ---
def main():
    parser = argparse.ArgumentParser(description="Run pose estimation inference on test images")
    
    # NOTE: Using --model_name to be consistent with train.py dispatcher
    parser.add_argument(
        "--model_name", 
        default=DEFAULT_MODEL_NAME,
        help="Model name (yolov8s-pose or dinov2_vits14)."
    )
    parser.add_argument(
        "--device", 
        default="cuda:0",
        help="Device for inference (e.g., '0' for GPU, 'cpu' for CPU)."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF_THRESHOLD,
        help="Confidence threshold for detections."
    )
    
    args = parser.parse_args()
    kwargs = vars(args)
    
    success = False
    
    if args.model_name.startswith("yolov"):
        from ..pipelines.yolo_pipeline import run_yolo_inference
        success = run_yolo_inference(**kwargs)
        
    elif args.model_name.startswith("dinov2"):
        from ..pipelines.dino_pipeline import run_dino_inference
        success = run_dino_inference(**kwargs)
        
    else:
        print(f"❌ Unknown model: {args.model_name}")
        sys.exit(1)

    if not success:
        sys.exit(2)


if __name__ == "__main__":
    main()