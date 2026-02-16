import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parent
ANNOTATIONS_PATH = PROJECT_ROOT / "data/main_dataset/annotations.json"
FRAMES_DIR = PROJECT_ROOT / "data/main_dataset/frames"
OUTPUT_VIDEO = PROJECT_ROOT / "verification_manual_fix.mp4"

# The substrings to identify manual frames
TARGET_VIDEOS = ["video12", "video13"]

# NOSE-FIRST SKELETON (For Visualization)
# Indices: 0:Nose, 1:Neck, 2:LSho, 3:RSho, 4:LElb, 5:RElb, 6:LWri, 7:RWri, 
#          8:LHip, 9:RHip, 10:LKnee, 11:RKnee, 12:LAnk, 13:RAnk
SKELETON_CONNECTIONS = [
    (0, 1),           # Nose -> Neck
    (1, 2), (1, 3),   # Neck -> Shoulders
    (2, 4), (4, 6),   # Left Arm
    (3, 5), (5, 7),   # Right Arm
    (2, 8), (3, 9),   # Torso
    (8, 9),           # Hips
    (8, 10), (10, 12),# Left Leg
    (9, 11), (11, 13) # Right Leg
]

def reorder_shoulder_to_nose(keypoints):
    """
    Maps Old Manual Format (Shoulder-First) to Target Format (Nose-First)
    """
    # Group into (x, y, v)
    kps = [keypoints[i:i+3] for i in range(0, len(keypoints), 3)]
    
    if len(kps) != 14:
        return keypoints # Safety return

    # OLD INDICES (Based on your YAML):
    # 0:LS, 1:RS, 2:LH, 3:RH, 4:LK, 5:RK, 6:LA, 7:RA, 
    # 8:LE, 9:LW, 10:RE, 11:RW, 12:Neck, 13:Nose

    # TARGET INDICES (ViTPose/Nose-First):
    # 0:Nose, 1:Neck, 2:LS, 3:RS, 4:LE, 5:RE, 6:LW, 7:RW, 
    # 8:LH, 9:RH, 10:LK, 11:RK, 12:LA, 13:RA

    new_order = [
        kps[13], kps[12], # Nose, Neck
        kps[0], kps[1],   # LSho, RSho
        kps[8], kps[10],  # LElb, RElb (Old 8=LE, 10=RE)
        kps[9], kps[11],  # LWri, RWri (Old 9=LW, 11=RW)
        kps[2], kps[3],   # LHip, RHip
        kps[4], kps[5],   # LKnee, RKnee
        kps[6], kps[7]    # LAnk, RAnk
    ]
    
    return [val for sublist in new_order for val in sublist]

def draw_skeleton(img, kpts, category_id):
    """Draws 14-point Nose-First skeleton"""
    # Colors (BGR)
    color = (0, 0, 255) if category_id == 1 else (255, 0, 0) # Red vs Blue
    
    # Reshape
    points = [(int(kpts[i]), int(kpts[i+1]), int(kpts[i+2])) for i in range(0, len(kpts), 3)]
    
    # Draw Lines
    for idx1, idx2 in SKELETON_CONNECTIONS:
        if idx1 < len(points) and idx2 < len(points):
            p1, p2 = points[idx1], points[idx2]
            if p1[2] > 0 and p2[2] > 0: # Check visibility
                cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]), (0, 255, 255), 2)

    # Draw Points (Nose is 0)
    for i, (x, y, v) in enumerate(points):
        if v > 0:
            c = (0, 255, 0) if i == 0 else color # Green for Nose
            cv2.circle(img, (x, y), 4, c, -1)
            
    return img

def main():
    print(f"📂 Loading annotations from: {ANNOTATIONS_PATH}")
    if not ANNOTATIONS_PATH.exists():
        print("❌ Annotation file not found!")
        return

    with open(ANNOTATIONS_PATH, 'r') as f:
        data = json.load(f)

    # 1. Identify Target Images
    print("🔍 Scanning for video12 and video13 frames...")
    target_img_ids = set()
    img_id_to_filename = {}
    
    for img in data['images']:
        fname = img['file_name']
        if any(v in fname for v in TARGET_VIDEOS):
            target_img_ids.add(img['id'])
            img_id_to_filename[img['id']] = fname

    print(f"   Found {len(target_img_ids)} manual frames to fix.")

    if len(target_img_ids) == 0:
        print("❌ No matching frames found. Check filter logic.")
        return

    # 2. Fix Annotations & Prepare for Video
    fixed_annotations_map = {} # image_id -> list of annotations
    
    print("🛠️  Fixing skeleton structure...")
    for ann in data['annotations']:
        if ann['image_id'] in target_img_ids:
            # FIX: Reorder Keypoints
            ann['keypoints'] = reorder_shoulder_to_nose(ann['keypoints'])
            
            # Store for visualization
            if ann['image_id'] not in fixed_annotations_map:
                fixed_annotations_map[ann['image_id']] = []
            fixed_annotations_map[ann['image_id']].append(ann)

    # 3. Generate Verification Video
    print(f"🎥 Generating verification video: {OUTPUT_VIDEO}")
    
    # Sort images by name to simulate video playback
    sorted_img_ids = sorted(target_img_ids, key=lambda x: img_id_to_filename[x])
    
    # Read first frame to get size
    first_frame_path = FRAMES_DIR / img_id_to_filename[sorted_img_ids[0]]
    frame0 = cv2.imread(str(first_frame_path))
    if frame0 is None:
        print(f"❌ Could not read image: {first_frame_path}")
        return
    
    h, w, _ = frame0.shape
    out = cv2.VideoWriter(str(OUTPUT_VIDEO), cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
    
    # Limit to first 300 frames for speed, or remove slicing to check all
    for img_id in tqdm(sorted_img_ids, desc="Rendering frames"): 
        fname = img_id_to_filename[img_id]
        img_path = FRAMES_DIR / fname
        
        frame = cv2.imread(str(img_path))
        if frame is None: continue
        
        # Draw all fixed annotations for this frame
        anns = fixed_annotations_map.get(img_id, [])
        for ann in anns:
            frame = draw_skeleton(frame, ann['keypoints'], ann['category_id'])
            
        # Add label
        cv2.putText(frame, f"{fname} (Fixed)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(frame)
        
    out.release()
    print("✅ Video saved.")

    # 4. Merge Back
    print("\n" + "="*50)
    print("⚠️  CHECK THE VIDEO NOW!")
    print("If the skeletons look correct (Green dot on Nose), type 'yes' to save changes.")
    print("="*50)
    
    choice = input("Overwrite annotations.json? [yes/no]: ").strip().lower()
    
    if choice == 'yes':
        # Update Metadata Categories to match Nose-First (just in case)
        data['categories'] = [
            {
                "id": 1, "name": "boxer_red", "supercategory": "person",
                "keypoints": ["nose", "neck", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"],
                "skeleton": [[0, 1], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [2, 8], [3, 9], [8, 9], [8, 10], [9, 11], [10, 12], [11, 13]]
            },
            {
                "id": 2, "name": "boxer_blue", "supercategory": "person",
                "keypoints": ["nose", "neck", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"],
                "skeleton": [[0, 1], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [2, 8], [3, 9], [8, 9], [8, 10], [9, 11], [10, 12], [11, 13]]
            }
        ]

        # Save
        with open(ANNOTATIONS_PATH, 'w') as f:
            json.dump(data, f)
        print("✅ annotations.json updated successfully!")
    else:
        print("❌ Changes discarded.")

def verify_vitpose_frames():
    print(f"📂 Loading annotations from: {ANNOTATIONS_PATH}")
    if not ANNOTATIONS_PATH.exists():
        print("❌ Annotation file not found!")
        return

    with open(ANNOTATIONS_PATH, 'r') as f:
        data = json.load(f)

    # 1. Filter for ViTPose frames (Exclude video12 and video13)
    vitpose_img_ids = []
    img_id_to_filename = {}
    
    print("🔍 Filtering for ViTPose frames (excluding video12/13)...")
    for img in data['images']:
        fname = img['file_name']
        # Only keep if NOT video12 AND NOT video13
        if "video12" not in fname and "video13" not in fname:
            vitpose_img_ids.append(img['id'])
            img_id_to_filename[img['id']] = fname

    print(f"   Found {len(vitpose_img_ids)} ViTPose frames.")

    if not vitpose_img_ids:
        print("❌ No ViTPose frames found.")
        return

    # 2. Select 10 Random Samples
    samples = random.sample(vitpose_img_ids, min(10, len(vitpose_img_ids)))
    
    # Map image IDs to their annotations
    ann_map = {}
    for ann in data['annotations']:
        if ann['image_id'] in samples:
            if ann['image_id'] not in ann_map:
                ann_map[ann['image_id']] = []
            ann_map[ann['image_id']].append(ann)

    # 3. Visualize
    plt.figure(figsize=(20, 8))
    print("📊 Generating plot...")

    for idx, img_id in enumerate(samples):
        fname = img_id_to_filename[img_id]
        img_path = FRAMES_DIR / fname
        
        # Load Image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️ Could not read {fname}")
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Draw Annotations (Nose-First Structure)
        if img_id in ann_map:
            for ann in ann_map[img_id]:
                # We assume these are ALREADY Nose-First (from ViTPose)
                # So we just pass the keypoints directly to the drawer
                img = draw_skeleton(img, ann['keypoints'], ann['category_id'])

        # Plot
        plt.subplot(2, 5, idx + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"ID: {img_id}\n{fname}", fontsize=8)

    plt.tight_layout()
    plt.show() 
    print("✅ Displayed 10 random ViTPose samples.")

if __name__ == "__main__":
    verify_vitpose_frames()