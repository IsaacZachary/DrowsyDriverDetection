import os
import json
import shutil

# Paths
DATASET_PATH = "dataset/classification_frames"
ANNOTATIONS_FILE = os.path.join(DATASET_PATH, "annotations_all.json")

# Load annotations
with open(ANNOTATIONS_FILE, "r") as file:
    annotations = json.load(file)

# Output folders
categories = ["alert", "microsleep", "yawning"]
for category in categories:
    os.makedirs(os.path.join(DATASET_PATH, category), exist_ok=True)

# Process each image
for filename, data in annotations.items():
    state = data["driver_state"].lower()  # Normalize case
    if state in categories:
        src = os.path.join(DATASET_PATH, filename)
        dst = os.path.join(DATASET_PATH, state, filename)
        
        # Move file if it exists
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"Moved {filename} → {state}/")

print("✅ Images grouped successfully!")
