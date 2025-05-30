# Full pipeline 
import os
import subprocess

def run_script(path):
    print(f"\n🚀 Running: {path}")
    result = subprocess.run(["python", path], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("⚠️ Error:\n", result.stderr)

if __name__ == "__main__":
    print("🧠 Starting full ML pipeline...")

    # Step 1: Data Cleaning
    run_script("src/preprocessing/clean.py")

    # Step 2: Feature Selection
    run_script("src/feature_engineering/select_features.py")

    # Step 3: Model Training
    run_script("src/training/train.py")

    # Step 4: Model Testing
    run_script("src/testing/test_model.py")

    print("\n✅ Pipeline completed successfully!")
