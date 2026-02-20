#!/usr/bin/env python3
"""
Final Dataset Cleaner (MobileNetV2 + Strict 1-Boxer Enforcement)

This script:
1. Loads tracked annotations from `yolo_annotations_tracked.json`
2. Evaluates every bounding box in every frame using the trained MobileNetV2
3. Enforces the strict 1-Boxer rule (highest confidence wins per frame)
4. Deletes all poster annotations
5. Strips the temporary `track_id` metadata
6. Saves the final, pristine dataset to `yolo_annotations.json`
"""

import json
import cv2
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models

# ============================================================================
# PROJECT ROOT DETECTION
# ============================================================================

def get_project_root() -> Path:
    current_path = Path(__file__).resolve().parent
    for parent in [current_path] + list(current_path.parents):
        if (parent / "data").exists() or (parent / ".git").exists():
            return parent
    return Path.cwd()

PROJECT_ROOT = get_project_root()

# ============================================================================
# CONFIGURATION
# ============================================================================

TARGET_DIR = PROJECT_ROOT / "data" / "DATASET" / "SixClassBoxingVIDataset" / "V1"
INPUT_JSON = TARGET_DIR / "yolo_annotations_tracked.json"
OUTPUT_JSON = TARGET_DIR / "yolo_annotations.json"
FRAMES_DIR = TARGET_DIR / "frames"
MODELS_DIR = PROJECT_ROOT / "models"
CLASSIFIER_PATH = MODELS_DIR / "poster_classifier.pth"

# ============================================================================
# MOBILENET CLASSIFIER INTEGRATION
# ============================================================================

def load_classifier(device: torch.device) -> nn.Module:
    print(f"\n⚙️  Loading MobileNetV2 Classifier...")
    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(f"❌ Classifier weights not found at {CLASSIFIER_PATH}")
        
    model = models.mobilenet_v2()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device, weights_only=True))
    model.eval().to(device)
    print("✅ Classifier loaded successfully")
    return model

def get_boxer_score(img, bbox, classifier, transform, device) -> float:
    """Crops the bounding box and returns the probability of it being a Boxer."""
    x, y, w, h = map(int, bbox)
    
    # Add padding to match training data
    pad = 20
    y1, y2 = max(0, y-pad), min(img.shape[0], y+h+pad)
    x1, x2 = max(0, x-pad), min(img.shape[1], x+w+pad)
    crop = img[y1:y2, x1:x2]
    
    if crop.size == 0:
        return 0.0
        
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = classifier(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        # PyTorch ImageFolder sorts alphabetically: 0 = boxer, 1 = poster
        return probabilities[0].item()

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print(f"\n{'='*80}")
    print("🧹 FINAL DATASET CLEANER (STRICT 1-BOXER ENFORCEMENT)")
    print(f"{'='*80}\n")
    
    if not INPUT_JSON.exists():
        print(f"❌ Input file not found: {INPUT_JSON}")
        return

    # 1. Setup PyTorch Device & Transform
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 2. Load the trained classifier
    try:
        classifier = load_classifier(device)
    except FileNotFoundError as e:
        print(e)
        return

    # 3. Load tracked annotations
    print(f"\n📄 Loading tracked annotations from {INPUT_JSON.name}...")
    with open(INPUT_JSON, 'r') as f:
        data = json.load(f)
        
    annotations = data.get('annotations', [])
    initial_count = len(annotations)
    
    # Group annotations by frame
    frames_to_anns = defaultdict(list)
    for ann in annotations:
        frames_to_anns[ann['frame_number']].append(ann)
        
    print(f"📊 Found {initial_count} total bounding boxes across {len(frames_to_anns)} frames.")

    # 4. Filter frames (The Battle Royale)
    print(f"\n🥊 Refereeing frames (Highest confidence Boxer wins)...")
    
    final_cleaned_annotations = []
    missing_frames = 0
    
    for frame_num in tqdm(sorted(frames_to_anns.keys()), desc="Cleaning"):
        anns = frames_to_anns[frame_num]
        frame_path = FRAMES_DIR / f"frame_{frame_num:06d}.jpg"
        
        if not frame_path.exists():
            missing_frames += 1
            continue
            
        img = cv2.imread(str(frame_path))
        if img is None:
            missing_frames += 1
            continue
            
        # Score every annotation in the frame
        scored_anns = []
        for ann in anns:
            score = get_boxer_score(img, ann['bbox'], classifier, transform, device)
            scored_anns.append((score, ann))
            
        # The annotation with the highest boxer score wins
        best_score, winning_ann = max(scored_anns, key=lambda x: x[0])
        
        # Strip the temporary tracking metadata
        if 'track_id' in winning_ann:
            del winning_ann['track_id']
            
        final_cleaned_annotations.append(winning_ann)

    # 5. Re-index the IDs to be sequential (COCO standard)
    for i, ann in enumerate(final_cleaned_annotations):
        ann['id'] = i + 1

    # 6. Save the cleaned dataset
    print(f"\n💾 Saving cleaned annotations to {OUTPUT_JSON.name}...")
    
    data['info']['description'] = "YOLO 14-point skeleton (Cleaned: Strict 1-Boxer)"
    data['info']['version'] = "2.0"
    data['annotations'] = final_cleaned_annotations
    
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(data, f, indent=2)
        
    # 7. Print Summary
    removed_count = initial_count - len(final_cleaned_annotations)
    print("\n" + "="*80)
    print("CLEANING COMPLETE")
    print("="*80)
    print(f"📉 Initial Bounding Boxes: {initial_count}")
    print(f"🗑️  Posters Removed:        {removed_count}")
    print(f"📈 Final Boxers Kept:       {len(final_cleaned_annotations)}")
    if missing_frames > 0:
        print(f"⚠️  Missing/Corrupted Images: {missing_frames}")
    print(f"✅ Success! Your pristine dataset is ready at: {OUTPUT_JSON.name}")

if __name__ == "__main__":
    main()