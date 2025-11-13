import sys
import importlib
import subprocess
from pathlib import Path

def menu():
    print("\n==== Boxer Pose Estimation Pipeline ====\n")
    print("1. Extract frames from video")
    print("2. Merge COCO annotation datasets")
    print("3. Split dataset into train/val/test")
    print("4. Train model")
    print("5. Run inference")
    print("6. Visualize results")
    print("7. Rename COCO frame annotations")
    print("8: Cleanup/Convert Annotations")
    print("0. Exit")
    choice = input("\nEnter your choice: ").strip()
    return choice

# New function to handle model selection before calling training/inference
def run_model_selection(script_name):
    print(f"\n--- Model Selection for {script_name.title()} ---")
    print("1. YOLOv8s-Pose")
    print("2. YOLOv8m-Pose")
    print("3. DinoV2-ViTS14 (Top-Down)")
    model_choice = input("Select model (1 or 2): ").strip()
    
    if model_choice == "1":
        model_name = "yolov8s-pose"
    elif model_choice == "2":
        model_name = "yolov8m-pose"
    elif model_choice == "3":
        model_name = "dinov2_vits14" # This is our new model identifier
    else:
        print("❌ Invalid model choice.")
        return

    # Call the target script using subprocess, passing the model name
    # script_path = PROJECT_ROOT / "src" / "training" / f"{script_name}.py"
    
    # We must pass the model name using the '--model' argument
    command = [sys.executable, "-m", f"src.training.{script_name}", "--model_name", model_name]
    
    print(f"\n🚀 Starting {script_name} for {model_name}...")
    try:
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Pipeline failed for {model_name}. Error: {e}")

def run_script(module_name, func_name="main"):
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, func_name):
            getattr(module, func_name)()
        else:
            print(f"❌ {module_name} does not have function '{func_name}'")
    except Exception as e:
        print(f"❌ Error running {module_name}: {e}")

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    while True:
        choice = menu()

        if choice == "1":
            run_script("src.data_processing.extract_frames")
        elif choice == "2":
            run_script("src.data_processing.merge_datasets")
        elif choice == "3":
            run_script("src.data_processing.split_dataset")
        elif choice == "4":
            run_model_selection("train") # Calls train.py
        elif choice == "5":
            run_model_selection("inference") # Calls inference.py
        elif choice == "6":
            run_script("src.utils.visualize")
        elif choice == "7":
            run_script("src.data_processing.rename_frames")
        elif choice == "8":
            run_script("src.data_processing.annotations_cleaner")
        elif choice == "0":
            print("👋 Exiting... Bye!")
            sys.exit(0)
        else:
            print("❌ Invalid choice, try again.")
