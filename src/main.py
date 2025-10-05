import sys
import importlib

def menu():
    print("\n==== Boxer Pose Estimation Pipeline ====\n")
    print("1. Extract frames from video")
    print("2. Merge COCO annotation datasets")
    print("3. Split dataset into train/val/test")
    print("4. Train model")
    print("5. Run inference")
    print("6. Visualize results")
    print("0. Exit")
    choice = input("\nEnter your choice: ").strip()
    return choice

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
    while True:
        choice = menu()

        if choice == "1":
            run_script("src.data_processing.extract_frames")
        elif choice == "2":
            run_script("src.data_processing.merge_datasets")
        elif choice == "3":
            run_script("src.data_processing.split_dataset")
        elif choice == "4":
            run_script("src.training.train")
        elif choice == "5":
            run_script("src.training.inference")
        elif choice == "6":
            run_script("src.utils.visualize")
        elif choice == "0":
            print("👋 Exiting... Bye!")
            sys.exit(0)
        else:
            print("❌ Invalid choice, try again.")
