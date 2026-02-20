import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path

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
# PATH CONFIGURATION
# ============================================================================
# Pointing exactly to the 'train' folder shown in your screenshot
TRAIN_DIR = PROJECT_ROOT / "data" / "DATASET" / "SixClassBoxingVIDataset" / "V1" / "train"
MODELS_DIR = PROJECT_ROOT / "models"

def train():
    # Sanity check for the dataset folder
    if not TRAIN_DIR.exists():
        print(f"❌ Train directory not found: {TRAIN_DIR}")
        return

    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_save_path = MODELS_DIR / 'poster_classifier.pth'

    # 1. Prepare Data (Resize to 224x224 and Normalize)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Point ImageFolder directly to the 'train' directory containing 'boxer' and 'poster'
    dataset = datasets.ImageFolder(str(TRAIN_DIR), transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print(f"📁 Loaded dataset from: {TRAIN_DIR.relative_to(PROJECT_ROOT)}")
    print(f"🏷️  Classes detected: {dataset.class_to_idx}") 

    # 2. Load Pretrained MobileNetV2 and modify for Binary Classification
    print("⚙️  Loading MobileNetV2 architecture...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    
    # Freeze the feature layers so we only train the classifier head
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace the final layer to output 2 classes instead of 1000
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"🚀 Using device: {device.type.upper()}")

    # 3. Train
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    print("\n🚀 Starting training...")
    for epoch in range(5):  # 5 epochs is usually enough for this
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"   Epoch {epoch+1}/5 - Loss: {running_loss/len(dataloader):.4f}")

    # Save the trained weights
    torch.save(model.state_dict(), str(model_save_path))
    print(f"\n✅ Model successfully saved to: {model_save_path.relative_to(PROJECT_ROOT)}")

if __name__ == "__main__":
    train()