import json
import os

# Automatically detect the project root (2 levels above this file)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
ANNOTATION_ROOT = os.path.join(PROJECT_ROOT, "data/raw/annotations")

def list_annotation_dirs():
    """List all directories under /data/raw/annotations."""
    if not os.path.exists(ANNOTATION_ROOT):
        print(f"⚠️ Annotation root not found at: {ANNOTATION_ROOT}")
        return None

    dirs = [d for d in os.listdir(ANNOTATION_ROOT) if os.path.isdir(os.path.join(ANNOTATION_ROOT, d))]
    if not dirs:
        print("❌ No annotation directories found!")
        return None

    print("\nAvailable annotation directories:")
    for idx, d in enumerate(dirs, 1):
        print(f"{idx}. {d}")

    while True:
        try:
            choice = int(input("\nEnter the number of the directory to process: "))
            if 1 <= choice <= len(dirs):
                return os.path.join(ANNOTATION_ROOT, dirs[choice - 1]), dirs[choice - 1]
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")


def fix_frame_names(ann_dir, video_name):
    """Fix frame naming in the annotations.json file using the directory name."""
    json_path = os.path.join(ann_dir, "annotations.json")
    if not os.path.exists(json_path):
        print(f"❌ No annotations.json found in {ann_dir}")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    for img in data.get("images", []):
        new_name = f"{video_name}_frame_{img['id']:06d}.png"
        img["file_name"] = new_name

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Updated frame names for '{video_name}' in {json_path}")


def main():
    result = list_annotation_dirs()
    if result:
        ann_dir, video_name = result
        fix_frame_names(ann_dir, video_name)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ An error occurred: {e}")
