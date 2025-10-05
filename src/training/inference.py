import argparse
from pathlib import Path
from ultralytics import YOLO
import sys
import torch

# Add src to path for imports
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from utils.results_to_coco import save_results_as_coco

DATA_PROCESSED = PROJECT / "data" / "processed"
IMAGES_TEST = DATA_PROCESSED / "images" / "test"
ANNOTS_PROCESSED = DATA_PROCESSED / "annotations"
OUT_TEST_JSON = ANNOTS_PROCESSED / "test.json"
MODELS_DIR = PROJECT / "models"

DEFAULT_MODEL_NAME = "yolov8s-pose"
DEFAULT_CONF_THRESHOLD = 0.2


def run_inference(model_name: str = DEFAULT_MODEL_NAME, device: str = "0", conf_threshold: float = DEFAULT_CONF_THRESHOLD):
    """
    Run inference on test images and save results as COCO JSON.
    
    Args:
        model_name: Name of the model directory in /models/
        device: Device to run inference on (e.g., '0', 'cpu')
        conf_threshold: Confidence threshold for detections
    """
    model_path = MODELS_DIR / model_name / "best.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading model from: {model_path}")
    model = YOLO(str(model_path))

    # Collect test images (sorted for consistency)
    image_paths = sorted([
        p for p in IMAGES_TEST.glob("*") 
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])
    
    if not image_paths:
        print(f"No test images found at: {IMAGES_TEST}")
        return

    print(f"Found {len(image_paths)} test images")
    print(f"Running inference with confidence threshold: {conf_threshold}")
    
    # Run inference on all images at once
    results = model.predict(
        source=[str(p) for p in image_paths],
        imgsz=640,
        save=False,
        device=device,
        conf=conf_threshold,
        verbose=True
    )

    # debug: make results a list and print quick summary
    results = list(results)
    print(f"[DEBUG] results length: {len(results)}, images: {len(image_paths)}")
    for i, r in enumerate(results, start=1):
        r_cpu = r.cpu()
        has_kp = hasattr(r_cpu, 'keypoints') and r_cpu.keypoints is not None
        kp_shape = None
        if has_kp:
            try:
                kp = r_cpu.keypoints.xy
                kp_shape = tuple(kp.shape)  # (num_people, n_keypoints, 2)
            except Exception as e:
                kp_shape = f"error:{e}"
        boxes_len = 0
        if hasattr(r_cpu, 'boxes') and r_cpu.boxes is not None:
            try:
                boxes_len = len(r_cpu.boxes.xyxy)
            except Exception:
                boxes_len = "?"
        print(f"[DEBUG] image {i}: file={image_paths[i-1].name}, has_keypoints={has_kp}, keypoints_shape={kp_shape}, boxes={boxes_len}")
        
    results = list(results)

    print(f"\nConverting results to COCO format...")
    
    # Convert results to COCO format and save
    save_results_as_coco(
        results=results,
        image_paths=image_paths,
        out_json=OUT_TEST_JSON,
        overwrite=True
    )

    print(f"✅ Inference complete! COCO annotations saved to: {OUT_TEST_JSON}")
    print(f"   - Total images: {len(image_paths)}")
    print(f"   - Total annotations: {sum(len(r.boxes) if hasattr(r, 'boxes') and r.boxes is not None else 0 for r in results)}")


def main():
    parser = argparse.ArgumentParser(description="Run pose estimation inference on test images")
    parser.add_argument(
        "--model", 
        default=DEFAULT_MODEL_NAME,
        help=f"Model name (directory in /models/). Default: {DEFAULT_MODEL_NAME}"
    )
    parser.add_argument(
        "--device", 
        default="0",
        help="Device for inference (e.g., '0' for GPU, 'cpu' for CPU). Default: 0"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF_THRESHOLD,
        help=f"Confidence threshold for detections. Default: {DEFAULT_CONF_THRESHOLD}"
    )
    
    args = parser.parse_args()
    
    run_inference(model_name=args.model, device=args.device, conf_threshold=args.conf)


if __name__ == "__main__":
    main()