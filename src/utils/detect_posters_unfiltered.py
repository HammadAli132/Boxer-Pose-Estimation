#!/usr/bin/env python3
"""
Robust Poster Detection Using MobileNetV2 Visual Classifier (Strict 1-Boxer Rule)

This script:
1. Loads frame ranges from annotations.json
2. Uses YOLO tracking (model.track()) to track poses across frames
3. Crops the bounding boxes of tracked objects
4. Passes crops through a trained MobileNetV2 classifier
5. Enforces STRICTLY 1 BOXER PER FRAME (The highest confidence bbox wins)
6. Visualizes results with RED=static (poster), GREEN=dynamic (boxer)
"""

import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
from collections import defaultdict
from PIL import Image
import random

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
ANNOTATIONS_PATH = TARGET_DIR / "annotations.json"
FRAMES_DIR = TARGET_DIR / "frames"
MODELS_DIR = PROJECT_ROOT / "models"
CLASSIFIER_PATH = MODELS_DIR / "poster_classifier.pth"

# ============================================================================
# SKELETON CONVERSION
# ============================================================================

def convert_kpts_coco17_to_custom14(kpts: np.ndarray, confs: np.ndarray) -> List[float]:
    """Convert COCO 17-point skeleton to custom 14-point skeleton"""
    l_sho, r_sho = kpts[5], kpts[6]
    neck_xy = (l_sho + r_sho) / 2.0
    neck_conf = min(confs[5], confs[6])

    custom_ordered_indices = [0, None, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    
    flat_list = []
    for idx in custom_ordered_indices:
        if idx is None:
            x, y = neck_xy
            c = neck_conf
        else:
            x, y = kpts[idx]
            c = confs[idx]
        
        # v = 2 if c > 0.4 else 0
        flat_list.extend([float(x), float(y), float(c)])
    
    return flat_list

# ============================================================================
# MOBILENET CLASSIFIER INTEGRATION
# ============================================================================

def load_classifier(device: torch.device) -> nn.Module:
    print(f"\n⚙️  Loading MobileNetV2 Classifier...")
    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(f"Classifier weights not found at {CLASSIFIER_PATH}")
        
    model = models.mobilenet_v2()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device, weights_only=True))
    model.eval().to(device)
    print("✅ Classifier loaded successfully")
    return model

def classify_track(track_history: List[Dict], classifier: nn.Module, 
                   transform: transforms.Compose, device: torch.device, 
                   num_samples: int = 5) -> float:
    """
    Passes a tracked object through MobileNet and averages its 'Boxer' probability.
    Returns: Average Boxer Probability (0.0 to 1.0)
    """
    step = max(1, len(track_history) // num_samples)
    samples = track_history[::step][:num_samples]
    
    boxer_probs = []
    
    for ann in samples:
        frame_path = FRAMES_DIR / f"frame_{ann['frame_number']:06d}.jpg"
        if not frame_path.exists(): continue
            
        img = cv2.imread(str(frame_path))
        if img is None: continue
            
        x, y, w, h = map(int, ann['bbox'])
        
        pad = 20
        y1, y2 = max(0, y-pad), min(img.shape[0], y+h+pad)
        x1, x2 = max(0, x-pad), min(img.shape[1], x+w+pad)
        crop = img[y1:y2, x1:x2]
        
        if crop.size == 0: continue
            
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = classifier(input_tensor)
            # Apply softmax to convert raw scores into percentages
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            # PyTorch ImageFolder sorts alphabetically: 0 = boxer, 1 = poster
            prob_boxer = probabilities[0].item()
            boxer_probs.append(prob_boxer)
                
    if not boxer_probs:
        return 0.0 # Failsafe
        
    # Return the average probability of being a boxer
    return sum(boxer_probs) / len(boxer_probs)

# ============================================================================
# YOLO TRACKING
# ============================================================================

def run_yolo_tracking(model, frame_paths: List[Path]) -> Dict[int, List[Dict]]:
    track_histories = defaultdict(list)
    print(f"\n🔍 Running YOLO tracking on {len(frame_paths)} frames...")
    
    for frame_path in tqdm(frame_paths, desc="Tracking"):
        frame_num = int(frame_path.stem.split("_")[1])
        
        results = model.track(str(frame_path), persist=True, verbose=False)
        result = results[0]
        
        if result.boxes is None or result.keypoints is None:
            continue
        
        if hasattr(result.boxes, 'id') and result.boxes.id is not None:
            track_ids = result.boxes.id.cpu().numpy().astype(int)
        else:
            continue 
        
        xyxy_boxes = result.boxes.xyxy.cpu().numpy()
        kpts_data = result.keypoints.data.cpu().numpy()
        
        for idx, track_id in enumerate(track_ids):
            x1, y1, x2, y2 = xyxy_boxes[idx]
            bbox_w, bbox_h = x2 - x1, y2 - y1
            bbox = [float(x1), float(y1), float(bbox_w), float(bbox_h)]
            area = float(bbox_w * bbox_h)
            
            person_kpts = kpts_data[idx]
            coords = person_kpts[:, :2]
            confs = person_kpts[:, 2]
            
            custom_kpts = convert_kpts_coco17_to_custom14(coords, confs)
            
            annotation = {
                "frame_number": frame_num,
                "track_id": int(track_id),
                "keypoints": custom_kpts,
                "bbox": bbox,
                "area": area,
                "num_keypoints": sum(1 for i in range(2, len(custom_kpts), 3) if custom_kpts[i] > 0)
            }
            track_histories[int(track_id)].append(annotation)
            
    return track_histories

# ============================================================================
# DATA MANAGEMENT & VISUALIZATION
# ============================================================================

def load_annotations_from_file(yolo_ann_path: Path) -> Dict[int, List[Dict]]:
    with open(yolo_ann_path, 'r') as f:
        yolo_data = json.load(f)
    
    track_histories = defaultdict(list)
    for ann in yolo_data.get('annotations', []):
        if 'track_id' in ann:
            track_id = ann['track_id']
            track_histories[track_id].append(ann)
        else:
            track_id = ann.get('id', len(track_histories))
            track_histories[track_id].append(ann)
    return track_histories

def save_tracked_annotations(track_histories: Dict[int, List[Dict]], output_path: Path):
    print(f"\n💾 Saving tracked annotations to {output_path.name}...")
    
    yolo_output = {
        "info": {"description": "YOLO 14-point skeleton tracked annotations", "version": "1.1"},
        "categories": [{"id": 1, "name": "person"}],
        "annotations": []
    }
    
    ann_id = 1
    for track_id, history in track_histories.items():
        for ann in history:
            annotation = {
                "id": ann_id,
                "frame_number": ann["frame_number"],
                "track_id": track_id,
                "category_id": 1,
                "keypoints": ann["keypoints"],
                "bbox": ann["bbox"],
                "area": ann["area"],
                "num_keypoints": ann["num_keypoints"],
                "iscrowd": 0
            }
            yolo_output["annotations"].append(annotation)
            ann_id += 1
            
    with open(output_path, 'w') as f:
        json.dump(yolo_output, f, indent=2)
    print(f"✅ Saved {ann_id - 1} tracked annotations.")

def draw_annotation(frame: np.ndarray, ann: Dict, color: Tuple[int, int, int], 
                   label: str, thickness: int = 3):
    bbox = ann['bbox']
    keypoints = ann['keypoints']
    x, y, w, h = map(int, bbox)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    (tw, th), bl = cv2.getTextSize(label, font, font_scale, font_thickness)
    cv2.rectangle(frame, (x, y - th - bl - 10), (x + tw + 6, y - 2), color, -1)
    cv2.putText(frame, label, (x + 3, y - bl - 6), font, font_scale, 
                (255, 255, 255), font_thickness, cv2.LINE_AA)
    
    for i in range(0, len(keypoints), 3):
        kx, ky, v = keypoints[i], keypoints[i+1], keypoints[i+2]
        if v > 0:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 255), -1)
    return frame

def visualize_tracks(frames_dir: Path, track_histories: Dict[int, List[Dict]], 
                    track_classifications: Dict[int, Dict], frame_boxer_map: Dict[int, int],
                    sample_frames: List[int]):
    n_frames = len(sample_frames)
    cols = min(3, n_frames)
    rows = (n_frames + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(7*cols, 7*rows))
    if n_frames == 1: axes = [axes]
    else: axes = axes.flatten() if rows > 1 else axes
    
    fig.suptitle('Strict 1-Boxer Rule (GREEN=Highest Boxer Prob, RED=Poster)', 
                 fontsize=16, fontweight='bold')
    
    for idx, frame_num in enumerate(sample_frames):
        frame_path = frames_dir / f"frame_{frame_num:06d}.jpg"
        frame = cv2.imread(str(frame_path))
        if frame is None: continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        winner_track_id = frame_boxer_map.get(frame_num)
        
        for track_id, history in track_histories.items():
            for ann in history:
                if ann['frame_number'] == frame_num:
                    score = track_classifications[track_id]['boxer_score']
                    conf_str = f"{score * 100:.1f}%"
                    
                    if track_id == winner_track_id:
                        color = (0, 255, 0)
                        label = f"BOXER ({conf_str})"
                    else:
                        color = (255, 0, 0)
                        label = f"POSTER ({conf_str})"
                        
                    draw_annotation(frame, ann, color, label)
                    break
        
        axes[idx].imshow(frame)
        axes[idx].axis('off')
        axes[idx].set_title(f"Frame {frame_num}", fontsize=12)
    
    for idx in range(n_frames, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"\n{'='*80}")
    print("AI POSTER DETECTION - STRICT 1-BOXER ENFORCEMENT")
    print(f"{'='*80}\n")
    
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

    # 3. Check for existing tracks
    yolo_annotations_path = TARGET_DIR / "yolo_annotations_tracked.json"
    use_existing = False
    track_histories = None
    
    if yolo_annotations_path.exists():
        print(f"\n✅ Found existing yolo_annotations_tracked.json")
        response = input("💡 Use existing tracking data? (y/n): ")
        if response.lower() == 'y':
            use_existing = True
            track_histories = load_annotations_from_file(yolo_annotations_path)
            print(f"✅ Loaded {len(track_histories)} tracks")
    
    if not use_existing:
        print(f"📄 Loading annotations...")
        with open(ANNOTATIONS_PATH, 'r') as f:
            annotations_data = json.load(f)
        
        frame_ranges = [(a["start_frame"], a["end_frame"]) for a in annotations_data.get("annotations", []) 
                        if "start_frame" in a and "end_frame" in a]
        
        all_frames = set()
        for start, end in frame_ranges: all_frames.update(range(start, end + 1))
        
        frame_paths = [FRAMES_DIR / f"frame_{f:06d}.jpg" for f in sorted(all_frames) 
                       if (FRAMES_DIR / f"frame_{f:06d}.jpg").exists()]
        
        from ultralytics import YOLO
        yolo_model = YOLO(str(MODELS_DIR / "yolov11x-pose" / "yolo11x-pose.pt"))
        
        track_histories = run_yolo_tracking(yolo_model, frame_paths)
        save_tracked_annotations(track_histories, yolo_annotations_path)
    
    # 4. Score each track visually with MobileNet
    print(f"\n🖼️  Calculating Boxer probabilities with MobileNetV2...")
    track_classifications = {}
    
    for track_id, history in track_histories.items():
        boxer_score = classify_track(history, classifier, transform, device)
        track_classifications[track_id] = {
            "boxer_score": boxer_score,
            "track_length": len(history)
        }
        
    # 5. Enforce STRICT 1-BOXER Rule per Frame
    print(f"\n🥊 Resolving frames (Enforcing exactly 1 boxer per frame)...")
    frame_to_tracks = defaultdict(list)
    
    # Map each frame to the tracks present in it
    for track_id, history in track_histories.items():
        for ann in history:
            frame_to_tracks[ann['frame_number']].append(track_id)
            
    frame_boxer_map = {}
    
    # In each frame, the track with the highest Boxer Score wins
    for frame_num, tids in frame_to_tracks.items():
        # Find the track_id that has the highest 'boxer_score'
        winner_id = max(tids, key=lambda t: track_classifications[t]['boxer_score'])
        frame_boxer_map[frame_num] = winner_id
        
    print(f"✅ Guaranteed exactly 1 Boxer per frame across {len(frame_boxer_map)} frames.")
    
    # 6. Visualization
    print(f"\n🎨 Creating visualization...")
    
    tracked_frames = list(frame_to_tracks.keys())
    
    if len(tracked_frames) > 10:
        # Take 10 completely random frames every run
        sample_frames = sorted(random.sample(tracked_frames, 10))
    else:
        sample_frames = sorted(tracked_frames)
    
    visualize_tracks(FRAMES_DIR, track_histories, track_classifications, frame_boxer_map, sample_frames)

if __name__ == "__main__":
    main()