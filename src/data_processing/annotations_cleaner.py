import json
import os
from pathlib import Path

# Automatically detect the project root (2 levels above this file)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANNOTATION_ROOT = PROJECT_ROOT / "data" / "raw" / "annotations"


def list_annotation_dirs(prompt_text="Select an annotation directory:"):
    """List all directories under /data/raw/annotations and let user select one."""
    if not ANNOTATION_ROOT.exists():
        print(f"⚠️ Annotation root not found at: {ANNOTATION_ROOT}")
        return None

    dirs = [d for d in os.listdir(ANNOTATION_ROOT) if (ANNOTATION_ROOT / d).is_dir()]
    if not dirs:
        print("❌ No annotation directories found!")
        return None

    print(f"\n{prompt_text}")
    for idx, d in enumerate(dirs, 1):
        print(f"{idx}. {d}")
    print(f"{len(dirs) + 1}. Cancel")

    while True:
        try:
            choice = int(input("\nEnter your choice: "))
            if choice == len(dirs) + 1:
                return None
            if 1 <= choice <= len(dirs):
                selected_dir = dirs[choice - 1]
                return ANNOTATION_ROOT / selected_dir, selected_dir
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")


def fix_frame_names(ann_dir, video_name):
    """Fix frame naming in the annotations.json file using the directory name."""
    json_path = ann_dir / "annotations.json"
    if not json_path.exists():
        print(f"❌ No annotations.json found in {ann_dir}")
        return False

    with open(json_path, "r") as f:
        data = json.load(f)

    for img in data.get("images", []):
        new_name = f"{video_name}_frame_{img['id']:06d}.png"
        img["file_name"] = new_name

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Updated frame names for '{video_name}' in {json_path}")
    return True


def cleanup_categories(ann_dir):
    """
    Clean up categories in COCO annotations file:
    1. Remove all categories except 'person'
    2. Set person category id to 1
    3. Update all annotations to use category_id = 1
    """
    annotations_path = ann_dir / "annotations.json"
    if not annotations_path.exists():
        print(f"❌ No annotations.json found in {ann_dir}")
        return False
    
    # Load annotations
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    
    print(f"\n📊 Original data:")
    print(f"  - Categories: {len(data['categories'])}")
    print(f"  - Annotations: {len(data['annotations'])}")
    
    # Find the person category
    person_category = None
    for category in data['categories']:
        if category['name'] == 'person':
            person_category = category
            break
    
    if person_category is None:
        print("❌ 'person' category not found in annotations!")
        return False
    
    print(f"✅ Found person category: ID {person_category['id']}")
    
    # Store the original person category ID for mapping
    original_person_id = person_category['id']
    
    # Update person category ID to 1
    person_category['id'] = 1
    
    # Replace categories list with only the person category
    data['categories'] = [person_category]
    
    # Update all annotations to use category_id = 1
    updated_annotations = 0
    for annotation in data['annotations']:
        if annotation['category_id'] == original_person_id:
            annotation['category_id'] = 1
            updated_annotations += 1
    
    print(f"🔄 Updated {updated_annotations} annotations to category_id = 1")
    
    # Save cleaned annotations
    output_path = ann_dir / "annotations.json"
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Cleanup complete!")
    print(f"📁 Output: {output_path}")
    print(f"📊 Final data:")
    print(f"  - Categories: {len(data['categories'])}")
    print(f"  - Annotations: {len(data['annotations'])}")
    print(f"  - Person category ID: {data['categories'][0]['id']}")
    
    return True


def add_color_attribute(ann_dir, default_color="red"):
    """Add missing color attribute to annotations"""
    annotations_path = ann_dir / "annotations.json"
    if not annotations_path.exists():
        print(f"❌ No annotations.json found in {ann_dir}")
        return False
    
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    
    count = 0
    for annotation in data['annotations']:
        if 'attributes' not in annotation:
            annotation['attributes'] = {}
        
        # Add color attribute with default value
        if 'color' not in annotation['attributes']:
            annotation['attributes']['color'] = default_color
            count += 1
    
    with open(annotations_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Added color attribute to {count} annotations")
    return True


def verify_cleanup(ann_dir):
    """Verify that the cleanup was successful"""
    annotations_path = ann_dir / "annotations.json"
    if not annotations_path.exists():
        print(f"❌ No annotations.json found in {ann_dir}")
        return False
    
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    
    print(f"\n🔍 Verification:")
    print(f"  - Categories count: {len(data['categories'])}")
    
    if data['categories']:
        category = data['categories'][0]
        print(f"  - Category name: {category['name']}")
        print(f"  - Category ID: {category['id']}")
    
    # Check annotations
    unique_category_ids = set()
    for ann in data['annotations']:
        unique_category_ids.add(ann['category_id'])
    
    print(f"  - Unique category IDs in annotations: {unique_category_ids}")
    
    if unique_category_ids == {1}:
        print("✅ Cleanup successful! All annotations use category_id = 1")
        return True
    else:
        print("❌ Cleanup incomplete! Found other category IDs")
        return False


def convert_annotations(current_ann_dir, trained_ann_dir):
    """
    Convert current annotations to match the trained model's keypoint order and skeleton structure.
    """
    current_json_path = current_ann_dir / "annotations.json"
    trained_json_path = trained_ann_dir / "annotations.json"
    
    if not current_json_path.exists():
        print(f"❌ No annotations.json found in {current_ann_dir}")
        return False
    
    if not trained_json_path.exists():
        print(f"❌ No annotations.json found in {trained_ann_dir}")
        return False
    
    # Load your current annotations
    with open(current_json_path, 'r') as f:
        current_data = json.load(f)
    
    # Load the trained model's annotations (for reference format)
    with open(trained_json_path, 'r') as f:
        trained_data = json.load(f)
    
    # Define the keypoint mapping from your order to trained model's order
    # Your order: [nose, neck, left_shoulder, right_shoulder, left_hip, right_hip, left_elbow, left_wrist, right_elbow, right_wrist, left_knee, left_ankle, right_knee, right_ankle]
    # Trained order: [left_shoulder, right_shoulder, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle, left_elbow, left_wrist, right_elbow, right_wrist, neck, nose]
    
    keypoint_mapping = {
        0: 13,   # your_nose -> their_nose
        1: 12,   # your_neck -> their_neck
        2: 0,    # your_left_shoulder -> their_left_shoulder
        3: 1,    # your_right_shoulder -> their_right_shoulder
        4: 2,    # your_left_hip -> their_left_hip
        5: 3,    # your_right_hip -> their_right_hip
        6: 8,    # your_left_elbow -> their_left_elbow
        7: 9,    # your_left_wrist -> their_left_wrist
        8: 10,   # your_right_elbow -> their_right_elbow
        9: 11,   # your_right_wrist -> their_right_wrist
        10: 4,   # your_left_knee -> their_left_knee
        11: 6,   # your_left_ankle -> their_left_ankle
        12: 5,   # your_right_knee -> their_right_knee
        13: 7    # your_right_ankle -> their_right_ankle
    }
    
    # Create the converted data structure
    converted_data = current_data.copy()
    
    # Update categories to match trained model's format
    trained_person_category = None
    for category in trained_data['categories']:
        if category['name'] == 'person':
            trained_person_category = category
            break
    
    if trained_person_category is None:
        print("❌ Person category not found in trained annotations")
        return False
    
    # Replace categories with trained model's person category
    converted_data['categories'] = [trained_person_category]
    
    # Convert all annotations
    for annotation in converted_data['annotations']:
        if annotation['category_id'] != 1:
            continue
            
        # Get the current keypoints
        current_keypoints = annotation['keypoints']
        
        # Convert to list of (x, y, v) tuples
        kp_tuples = []
        for i in range(0, len(current_keypoints), 3):
            kp_tuples.append((
                current_keypoints[i],
                current_keypoints[i+1],
                current_keypoints[i+2]
            ))
        
        # Reorder keypoints according to mapping
        reordered_kp_tuples = [None] * 14
        for old_idx, new_idx in keypoint_mapping.items():
            reordered_kp_tuples[new_idx] = kp_tuples[old_idx]
        
        # Flatten back to [x, y, v, x, y, v, ...] format
        converted_keypoints = []
        for kp in reordered_kp_tuples:
            if kp is not None:
                converted_keypoints.extend([kp[0], kp[1], kp[2]])
            else:
                converted_keypoints.extend([0, 0, 0])
        
        # Update the annotation
        annotation['keypoints'] = converted_keypoints
    
    # Save the converted annotations
    output_json_path = current_ann_dir / "annotations.json"
    with open(output_json_path, 'w') as f:
        json.dump(converted_data, f, indent=2)
    
    print(f"\n✅ Conversion complete!")
    print(f"Output: {output_json_path}")
    print(f"Converted {len(converted_data['annotations'])} annotations")
    
    # Verify conversion
    verify_conversion(converted_data)
    return True


def verify_conversion(converted_data):
    """Verify that the conversion was successful"""
    print("\n🔍 Verification:")
    
    # Check categories
    person_cat = converted_data['categories'][0]
    print(f"Category: {person_cat['name']} (id: {person_cat['id']})")
    print(f"Keypoints: {person_cat['keypoints']}")
    print(f"Keypoint count: {len(person_cat['keypoints'])}")
    print(f"Skeleton connections: {len(person_cat['skeleton'])}")
    
    # Check first annotation
    if converted_data['annotations']:
        first_ann = converted_data['annotations'][0]
        print(f"First annotation keypoints: {len(first_ann['keypoints'])} values")
        print(f"Expected: {14 * 3} = 42 values")
        print(f"Actual: {len(first_ann['keypoints'])} values")
        
        if len(first_ann['keypoints']) == 42:
            print("✅ Keypoint count correct!")
        else:
            print("❌ Keypoint count incorrect!")

def convert_to_multi_class(ann_dir):
    """
    Convert single-class annotations with color attributes to multi-class annotations.
    Creates two categories: boxer_red (id=1) and boxer_blue (id=2).
    Updates annotations based on their color attribute and removes the color attribute.
    """
    annotations_path = ann_dir / "annotations.json"
    if not annotations_path.exists():
        print(f"❌ No annotations.json found in {ann_dir}")
        return False
    
    # Load annotations
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    
    print(f"\n📊 Original data:")
    print(f"  - Categories: {len(data['categories'])}")
    print(f"  - Annotations: {len(data['annotations'])}")
    
    # Find the person category
    person_category = None
    for category in data['categories']:
        if category['name'] == 'person':
            person_category = category.copy()
            break
    
    if person_category is None:
        print("❌ 'person' category not found in annotations!")
        return False
    
    print(f"✅ Found person category: ID {person_category['id']}")
    
    # Create two new categories
    boxer_red_category = person_category.copy()
    boxer_red_category['id'] = 1
    boxer_red_category['name'] = 'boxer_red'
    
    boxer_blue_category = person_category.copy()
    boxer_blue_category['id'] = 2
    boxer_blue_category['name'] = 'boxer_blue'
    
    # Replace categories list
    data['categories'] = [boxer_red_category, boxer_blue_category]
    
    print(f"\n✅ Created two categories:")
    print(f"  - boxer_red (id=1)")
    print(f"  - boxer_blue (id=2)")
    
    # Update annotations based on color attribute
    red_count = 0
    blue_count = 0
    no_color_count = 0
    
    for annotation in data['annotations']:
        # Check color attribute
        color = None
        if 'attributes' in annotation and 'color' in annotation['attributes']:
            color = annotation['attributes']['color'].lower()
        
        if color == 'red':
            annotation['category_id'] = 1
            red_count += 1
        elif color == 'blue':
            annotation['category_id'] = 2
            blue_count += 1
        else:
            # Default to red if no color specified
            annotation['category_id'] = 1
            no_color_count += 1
            print(f"⚠️  Warning: Annotation {annotation['id']} has no color attribute, defaulting to boxer_red")
        
        # Remove color attribute
        if 'attributes' in annotation and 'color' in annotation['attributes']:
            del annotation['attributes']['color']
    
    print(f"\n🔄 Updated annotations:")
    print(f"  - boxer_red: {red_count} annotations")
    print(f"  - boxer_blue: {blue_count} annotations")
    if no_color_count > 0:
        print(f"  - No color attribute (defaulted to red): {no_color_count} annotations")
    
    # Save as multi-label-annotations.json
    output_path = ann_dir / "multi-label-annotations.json"
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✅ Multi-class conversion complete!")
    print(f"📁 Output: {output_path}")
    print(f"📊 Final data:")
    print(f"  - Categories: {len(data['categories'])}")
    print(f"  - Total annotations: {len(data['annotations'])}")
    
    # Verify the conversion
    verify_multi_class(data)
    
    return True


def verify_multi_class(data):
    """Verify that the multi-class conversion was successful"""
    print("\n🔍 Verification:")
    
    # Check categories
    print(f"Categories ({len(data['categories'])}):")
    for cat in data['categories']:
        print(f"  - {cat['name']} (id={cat['id']})")
    
    # Count annotations per category
    category_counts = {}
    annotations_with_color = 0
    
    for ann in data['annotations']:
        cat_id = ann['category_id']
        category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
        
        # Check if color attribute still exists
        if 'attributes' in ann and 'color' in ann['attributes']:
            annotations_with_color += 1
    
    print(f"\nAnnotations per category:")
    for cat in data['categories']:
        count = category_counts.get(cat['id'], 0)
        print(f"  - {cat['name']}: {count} annotations")
    
    if annotations_with_color > 0:
        print(f"\n⚠️  Warning: {annotations_with_color} annotations still have color attribute!")
    else:
        print(f"\n✅ All color attributes removed successfully!")
    
    # Check for unexpected category IDs
    expected_ids = {1, 2}
    actual_ids = set(category_counts.keys())
    if actual_ids == expected_ids:
        print("✅ All annotations use valid category IDs (1 or 2)")
    else:
        print(f"❌ Found unexpected category IDs: {actual_ids - expected_ids}")

def show_menu():
    """Display the main menu"""
    print("\n" + "="*60)
    print("🔧 ANNOTATION CLEANER TOOLKIT")
    print("="*60)
    print("\n1. Fix Frame Names")
    print("2. Cleanup Categories (remove all except 'person')")
    print("3. Add Color Attribute")
    print("4. Convert Keypoint Order")
    print("5. Verify Cleanup")
    print("6. Convert to Multi-Class Annotations")
    print("0. Exit")
    print("\n" + "="*60)


def main():
    """Main function with menu loop"""
    while True:
        show_menu()
        
        try:
            choice = input("\nEnter your choice: ").strip()
            
            if choice == "0":
                print("\n👋 Goodbye!")
                break
            
            elif choice == "1":
                # Fix Frame Names
                result = list_annotation_dirs("Select directory to fix frame names:")
                if result:
                    ann_dir, video_name = result
                    fix_frame_names(ann_dir, video_name)
            
            elif choice == "2":
                # Cleanup Categories
                result = list_annotation_dirs("Select directory to cleanup categories:")
                if result:
                    ann_dir, _ = result
                    cleanup_categories(ann_dir)
            
            elif choice == "3":
                # Add Color Attribute
                result = list_annotation_dirs("Select directory to add color attribute:")
                if result:
                    ann_dir, _ = result
                    color = input("Enter default color (default: red): ").strip() or "red"
                    add_color_attribute(ann_dir, color)
            
            elif choice == "4":
                # Convert Keypoint Order
                print("\n--- SELECT SOURCE DIRECTORY (to be converted) ---")
                current_result = list_annotation_dirs("Select directory with annotations to convert:")
                if not current_result:
                    continue
                
                print("\n--- SELECT REFERENCE DIRECTORY (trained model format) ---")
                trained_result = list_annotation_dirs("Select directory with trained model annotations:")
                if not trained_result:
                    continue
                
                current_ann_dir, _ = current_result
                trained_ann_dir, _ = trained_result
                convert_annotations(current_ann_dir, trained_ann_dir)
            
            elif choice == "5":
                # Verify Cleanup
                result = list_annotation_dirs("Select directory to verify:")
                if result:
                    ann_dir, _ = result
                    verify_cleanup(ann_dir)
            
            elif choice == "6":
                # Convert to Multi-Class Annotations
                result = list_annotation_dirs("Select directory to convert to multi-class:")
                if result:
                    ann_dir, _ = result
                    convert_to_multi_class(ann_dir)
            
            else:
                print("❌ Invalid choice. Please try again.")
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()