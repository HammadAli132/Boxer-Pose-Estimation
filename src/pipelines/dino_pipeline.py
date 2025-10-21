import os
import sys
import shutil
import time
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import requests
import json
from PIL import Image
from typing import Tuple, List

# --- Path Setup ---
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
IMAGES_DIR = DATA_PROCESSED / "images"
ANNOTS_DIR = DATA_PROCESSED / "annotations"
MODELS_ROOT = PROJECT_ROOT / "models"
RUNS_ROOT = PROJECT_ROOT / "runs" 

# --- External Imports (Requires new file/module) ---
# NOTE: The DeconvolutionalHead must be placed in src/models/keypoint_heads.py
from models.keypoint_heads import DeconvolutionalHead 
# NOTE: Need a custom utility for inference-time COCO conversion (tensor to COCO)
from utils.tensor_to_coco import tensor_to_coco_json
from utils.heatmap_generation import generate_target_heatmaps


# --- CONSTANTS & CONFIGS ---
DINO_ARCH = "dinov2_vits14"
DINO_URL = f"https://dl.fbaipublicfiles.com/dinov2/{DINO_ARCH}/{DINO_ARCH}_pretrain.pth"
NUM_KEYPOINTS = 14
KEYPOINT_CHANNELS = NUM_KEYPOINTS # Output channels for heatmaps
HEAD_FINAL_RES = 64 # Final height/width of heatmap (e.g., 64x64)
CROP_SIZE = 256 # Input size for the cropped boxer image
DEFAULT_EPOCHS = 50
DEFAULT_BATCH = 4 # Small batch size for 8GB VRAM
IN_CHANNELS = 384 # DinoV2-ViTS14 feature dimension

# --- UTILITIES (Mirroring YOLO pipeline) ---

def ask_yes_no(prompt: str) -> bool:
    """Utility for yes/no questions in CLI."""
    while True:
        ans = input(f"{prompt} [y/n]: ").strip().lower()
        if ans in ("y", "yes"): return True
        if ans in ("n", "no"): return False
        print("Please enter 'y' or 'n'.")

def download_file(url: str, dest: Path):
    """Downloads weights (used for DinoV2 backbone)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"✅ Found local weights: {dest.name}. Skipping download.")
        return
    print(f"⬇️ Downloading weights from {url} to {dest}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("✅ Download complete.")

def ensure_dino_weights(model_name: str) -> Path:
    """Ensure pretrained DinoV2 weights are present."""
    model_dir = MODELS_ROOT / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    local_file = model_dir / f"{model_name}_backbone.pth"
    
    # NOTE: DinoV2 uses a simple download if not available locally
    download_file(DINO_URL, local_file)
    return local_file

# -----------------------
# DinoV2 Custom Dataset & Data Loader
# -----------------------

def generate_heatmap(keypoints: np.ndarray, bbox: np.ndarray, sigma: int = 2) -> np.ndarray:
    """
    Generates a Gaussian heatmap for a single person instance.
    NOTE: Requires mapping COCO keypoints/bbox/image dimensions to the fixed CROP_SIZE.
    """
    # Placeholder: In a real implementation, this converts KP coordinates 
    # relative to the bbox to Gaussian blobs in the HEAD_FINAL_RES space.
    heatmaps = np.zeros((NUM_KEYPOINTS, HEAD_FINAL_RES, HEAD_FINAL_RES), dtype=np.float32)
    # The actual implementation of this function is complex and involves:
    # 1. Scaling keypoints from (Image H, W) to (CROP_SIZE, CROP_SIZE).
    # 2. Scaling keypoints from (CROP_SIZE, CROP_SIZE) to (HEAD_FINAL_RES, HEAD_FINAL_RES).
    # 3. Drawing a Gaussian centered at the scaled keypoint on the map.
    return heatmaps


class DinoPoseDataset(Dataset):
    """Custom PyTorch Dataset for DinoV2 Top-Down Pose Estimation."""
    def __init__(self, split: str):
        # 1. Load COCO JSON for the split
        coco_path = ANNOTS_DIR / f"{split}.json"
        if not coco_path.exists():
            raise FileNotFoundError(f"Missing COCO JSON: {coco_path}")
        with open(coco_path, 'r') as f:
            self.coco = json.load(f)
            
        # 2. Pre-process and flatten all annotations (images are stored in images/train, images/val)
        self.image_map = {img['id']: img for img in self.coco['images']}
        self.split_dir = IMAGES_DIR / split
        
        # Filter annotations to create a list of single person instances
        self.instances = []
        for ann in self.coco['annotations']:
            if ann.get('num_keypoints', 0) > 0:
                self.instances.append(ann)
        
        if not self.instances:
            raise ValueError(f"No valid annotated instances found in {split} set.")
        
        print(f"Dataset loaded: {len(self.instances)} boxer instances for {split}.")

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        ann = self.instances[idx]
        img_info = self.image_map[ann['image_id']]
        
        # Load image
        img_path = self.split_dir / img_info['file_name']
        image = Image.open(img_path).convert('RGB')
        
        # 1. Crop image to boxer bounding box
        x, y, w, h = ann['bbox'] # BBox is [xmin, ymin, w, h] in COCO
        crop = image.crop((x, y, x + w, y + h))
        
        # 2. Preprocess: Resize crop to fixed input size and convert to tensor
        crop = crop.resize((CROP_SIZE, CROP_SIZE))
        # Simple Tensor conversion and normalization (In real code, use transforms)
        image_tensor = torch.from_numpy(np.array(crop)).permute(2, 0, 1).float() / 255.0 
        
        # 3. Generate Ground Truth Heatmaps
        # NOTE: This call relies on complex internal scaling logic (see utility placeholder)
        target_heatmaps = generate_target_heatmaps(
            keypoints_xyv=np.array(ann['keypoints'], dtype=np.float32), 
            bbox=np.array(ann['bbox'], dtype=np.float32),
            img_width=img_info['width'],
            img_height=img_info['height'],
        )
        target_tensor = torch.from_numpy(target_heatmaps)

        return image_tensor, target_tensor # (C, H, W) tensor, (14, H_out, W_out) tensor

# -----------------------
# DinoV2 Backbone Setup (Moved up)
# -----------------------

def load_dino_backbone(model_name: str) -> Tuple[nn.Module, int]:
    """Loads DinoV2-ViTS14 and freezes its weights."""
    print(f"⬇️ Loading DinoV2 backbone: {model_name}...")
    try:
        backbone = torch.hub.load('facebookresearch/dinov2', model_name, force_reload=False)
    except Exception as e:
        print(f"❌ Error loading DinoV2 from PyTorch Hub. Check internet/dependencies: {e}")
        sys.exit(1)

    # Freeze the backbone weights (CRITICAL)
    for param in backbone.parameters():
        param.requires_grad = False
        
    feature_dim = 384 # Hardcoding for ViTS14
    return backbone, feature_dim

# -----------------------
# DinoV2 Training and Inference Logic
# -----------------------

def setup_dino_model(model_name: str, resume: bool) -> nn.Module:
    """Combines frozen backbone and custom head, loading weights if resuming."""
    
    # 1. Load and Freeze Backbone
    # Ensure local weights exist (download if necessary)
    local_weights_path = ensure_dino_weights(model_name)
    
    # Load model state dict
    try:
        # DinoV2 does not expose a simple 'backbone' object via torch.load, 
        # so we rely on the Pytorch Hub call to load the correct structure.
        backbone, feature_dim = load_dino_backbone(model_name)
    except Exception as e:
        print(f"❌ Could not load DinoV2 via PyTorch Hub: {e}")
        sys.exit(1)

    # 2. Attach the Head (The trainable part)
    pose_head = DeconvolutionalHead(in_channels=feature_dim, num_keypoints=NUM_KEYPOINTS)
    
    # Load previously trained head weights if resuming
    checkpoint_target = MODELS_ROOT / model_name / "best_head.pt"
    if resume and checkpoint_target.exists():
        print(f"🔁 Resuming DinoV2 head from: {checkpoint_target}")
        pose_head.load_state_dict(torch.load(checkpoint_target))
    else:
        print("📦 Starting DinoV2 head training from scratch.")

    # 3. Combine models (backbone is frozen, head is trainable)
    model = nn.Sequential(backbone, pose_head)
    
    # Ensure only the head is truly trainable
    for name, param in model.named_parameters():
        if not name.startswith('0'): # '0' is the backbone in nn.Sequential
            param.requires_grad = True # Head is trainable
        else:
            param.requires_grad = False # Backbone is frozen
            
    return model


def run_dino_training(
    model_name: str, 
    epochs: int, 
    imgsz: int, # Ignored in favor of CROP_SIZE, but kept for signature
    batch: int,
    workers: int,
    no_resume: bool,
    **kwargs
) -> bool:
    print(f"\n--- DinoV2 Training Pipeline ({model_name}) ---")
    
    try:
        # Setup paths and environment
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = setup_dino_model(model_name, resume=not no_resume).to(device)
        
        # Setup Data Loaders
        train_dataset = DinoPoseDataset(split='train')
        val_dataset = DinoPoseDataset(split='val')
        train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=workers)
        val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False, num_workers=workers)

    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Data Setup Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Initialization Error: {e}")
        traceback.print_exc()
        return False
            
    # Setup Loss and Optimizer (only for the head)
    trainable_params = model[-1].parameters() 
    criterion = nn.MSELoss() # Standard for heatmap regression
    optimizer = optim.Adam(trainable_params, lr=1e-4)
    
    # Create new run directory (mirroring YOLO structure)
    run_number = 1
    while (RUNS_ROOT / model_name / f"run_{run_number}").exists():
        run_number += 1
    current_run_dir = RUNS_ROOT / model_name / f"run_{run_number}" / "train"
    current_run_dir.mkdir(parents=True)
    print(f"🧾 Logging this training to: {current_run_dir.parent}")
    
    best_loss = float('inf')
    
    # --- Training Loop ---
    print(f"🔥 Starting training for {epochs} epochs on {device}...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # Simulated/Actual Loop
        for images, target_heatmaps in train_loader:
            images, target_heatmaps = images.to(device), target_heatmaps.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, target_heatmaps)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)

        # --- Validation Step ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, target_heatmaps in val_loader:
                images, target_heatmaps = images.to(device), target_heatmaps.to(device)
                outputs = model(images)
                loss = criterion(outputs, target_heatmaps)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # --- Save Best Checkpoint ---
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            # Save the head's state dict
            torch.save(model[-1].state_dict(), MODELS_ROOT / model_name / "best_head.pt")
            print("⭐ New best model saved.")
            
        # Save last checkpoint (e.g., to run_dir for logs, and to models/ as last.pt)
        torch.save(model[-1].state_dict(), MODELS_ROOT / model_name / "last_head.pt")
        
    # --- Final Cleanup (Mirroring YOLO pipeline) ---
    # NOTE: Copy logs (results.csv, etc.) from run_dir/ to runs_root/model_name/run_N/train/ 
    # and perform old run cleanup, as done in yolo_pipeline.py.
    
    print("\n🎉 DinoV2 Head Training finished.")
    return True

# NOTE: The run_dino_inference function will be implemented fully once tensor_to_coco.py is done.
# It will use the new submodule code for detection.

def run_dino_inference(model_name: str, device: str, conf: float, **kwargs) -> bool:
    print("\n--- DinoV2 Inference Placeholder ---")
    print("Inference requires TF-DETR (submodule) and the tensor_to_coco.py utility.")
    return True