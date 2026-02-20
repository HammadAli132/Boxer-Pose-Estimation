#!/usr/bin/env python3
"""
Annotation Tree Extractor

This script:
1. Scans the specified DATASET directory for all `yolo_annotations.json` files.
2. Recreates the exact same folder structure in a new target directory.
3. Copies ONLY the JSON files over, leaving the heavy videos and frames behind.
"""

import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

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
# MAIN LOGIC
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Extract just the YOLO annotations into a mirrored directory structure.")
    parser.add_argument("--src", type=str, default="data/DATASET", help="Source dataset directory")
    parser.add_argument("--dst", type=str, default="data/ANNOTATIONS_EXPORT", help="Destination directory for the clean tree")
    args = parser.parse_args()

    src_dir = PROJECT_ROOT / args.src
    dst_dir = PROJECT_ROOT / args.dst

    print(f"\n{'='*80}")
    print("📂 ANNOTATION TREE EXTRACTOR")
    print(f"{'='*80}\n")

    if not src_dir.exists():
        print(f"❌ Source directory not found: {src_dir}")
        return

    # 1. Find all yolo_annotations.json files
    print(f"🔍 Scanning {src_dir.relative_to(PROJECT_ROOT)} for annotations...")
    json_files = list(src_dir.rglob("yolo_annotations.json"))
    
    if not json_files:
        print("❌ No 'yolo_annotations.json' files found in the source directory.")
        return
        
    print(f"✅ Found {len(json_files)} annotation files.\n")

    # 2. Mirror structure and copy
    dst_dir.mkdir(parents=True, exist_ok=True)
    print(f"💾 Copying files to {dst_dir.relative_to(PROJECT_ROOT)}...")

    success_count = 0
    
    for json_path in tqdm(json_files, desc="Copying", unit="file"):
        # Calculate the relative path from the source root
        # Example: SixClassBoxingVIDataset/V1/yolo_annotations.json
        rel_path = json_path.relative_to(src_dir)
        
        # Build the exact same path in the destination
        target_path = dst_dir / rel_path
        
        # Ensure the parent directories exist in the destination
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy the file (using copy2 to preserve timestamps/metadata)
        try:
            shutil.copy2(json_path, target_path)
            success_count += 1
        except Exception as e:
            print(f"\n❌ Failed to copy {rel_path}: {e}")

    # 3. Summary
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)
    print(f"📁 Source:      {src_dir.relative_to(PROJECT_ROOT)}")
    print(f"📁 Destination: {dst_dir.relative_to(PROJECT_ROOT)}")
    print(f"✅ Copied:      {success_count}/{len(json_files)} files successfully.")

if __name__ == "__main__":
    main()