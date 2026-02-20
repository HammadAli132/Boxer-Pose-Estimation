import cv2
import json
import random
from pathlib import Path

def get_project_root() -> Path:
    current_path = Path(__file__).resolve().parent
    for parent in [current_path] + list(current_path.parents):
        if (parent / "data").exists() or (parent / ".git").exists():
            return parent
    return Path.cwd()

PROJECT_ROOT = get_project_root()

def extract_crops():
    frames_dir = PROJECT_ROOT / "data" / "DATASET" / "SixClassBoxingVIDataset" / "V1" / "frames"
    yolo_json = PROJECT_ROOT / "data" / "DATASET" / "SixClassBoxingVIDataset" / "V1" / "yolo_annotations.json"
    output_dir = PROJECT_ROOT / "data" / "DATASET" / "SixClassBoxingVIDataset" / "V1" / "raw_crops"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(yolo_json, 'r') as f:
        data = json.load(f)

    annotations = data.get('annotations', [])
    
    if not annotations:
        print("❌ No annotations found in the JSON file.")
        return

    # Randomly sample 1000 annotations (or all of them if less than 1000 exist)
    num_samples = min(1000, len(annotations))
    sampled_annotations = random.sample(annotations, num_samples)

    print(f"🎲 Randomly selected {num_samples} annotations out of {len(annotations)}. Extracting...")

    count = 0
    # Loop through the randomly sampled annotations
    for ann in sampled_annotations:
        frame_num = ann['frame_number']
        frame_path = frames_dir / f"frame_{frame_num:06d}.jpg"
        
        if not frame_path.exists(): 
            continue
            
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
            
        x, y, w, h = map(int, ann['bbox'])
        
        # Add padding to the crop
        pad = 20
        y1, y2 = max(0, y-pad), min(img.shape[0], y+h+pad)
        x1, x2 = max(0, x-pad), min(img.shape[1], x+w+pad)
        
        crop = img[y1:y2, x1:x2]
        
        if crop.size > 0:
            # Added frame number to filename for easier debugging later
            filename = f"crop_{count:04d}_frame_{frame_num}_track_{ann.get('track_id', 'none')}.jpg"
            cv2.imwrite(str(output_dir / filename), crop)
            count += 1

    print(f"✅ Extracted {count} random crops. Now sort them manually!")

if __name__ == "__main__":
    # Optional: Set a seed if you want the "randomness" to be reproducible every time you run it
    # random.seed(42) 
    extract_crops()