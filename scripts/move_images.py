import os
import json
import shutil
import glob

# Paths
dataset_path = r"C:\Users\iZach\Documents\Zachh\DrowsyDriverDetection\dataset\classification_frames"
annotations_file = os.path.join(dataset_path, "annotations_all.json")

# Load JSON
with open(annotations_file, "r") as f:
    annotations = json.load(f)

# Ensure category folders exist inside classification_frames
categories = ["alert", "microsleep", "yawning"]
for category in categories:
    os.makedirs(os.path.join(dataset_path, category), exist_ok=True)

# Move images into respective folders
missing_files = []
for image_rel_path, data in annotations.items():
    driver_state = data["driver_state"]
    
    if driver_state in categories:
        # Extract filename only
        filename = os.path.basename(image_rel_path)

        # Search recursively for the correct image
        possible_matches = glob.glob(os.path.join(dataset_path, "**", filename), recursive=True)
        
        if possible_matches:
            source_path = possible_matches[0]  # Pick the first match
        else:
            missing_files.append(image_rel_path)
            print(f"❌ File not found: {image_rel_path}")
            continue  # Skip to next file
        
        # Move to category folder
        destination_folder = os.path.join(dataset_path, driver_state)
        destination_path = os.path.join(destination_folder, filename)
        
        shutil.move(source_path, destination_path)
        print(f"✅ Moved: {source_path} -> {destination_path}")

# Print missing files at the end
if missing_files:
    print("\n🚨 Missing Files List:")
    for file in missing_files:
        print(file)
