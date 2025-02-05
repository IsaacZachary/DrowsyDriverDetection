import os
import shutil
import json
import random

# Base directory containing images
base_dir = "dataset"

# Directories for train, test, and validation sets
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")
val_dir = os.path.join(base_dir, "validation")

# Create directories if they don't exist
for folder in [train_dir, test_dir, val_dir]:
    os.makedirs(folder, exist_ok=True)

# Load the annotations file
annotations_file = os.path.join(base_dir, "classification_frames", "annotations_train.json")
with open(annotations_file, "r") as f:
    annotations = json.load(f)  # Dictionary of images and metadata

# Convert dictionary to list of file paths
image_paths = list(annotations.keys())

# Shuffle for randomness
random.shuffle(image_paths)

# Split into train (80%), test (10%), validation (10%)
train_split = int(0.8 * len(image_paths))
test_split = int(0.9 * len(image_paths))  # 80% train, next 10% test, remaining 10% validation

train_images = image_paths[:train_split]
test_images = image_paths[train_split:test_split]
val_images = image_paths[test_split:]

# Function to move files
def move_files(image_list, target_dir):
    for image_path in image_list:
        relative_path = image_path.lstrip("./")  # Fix path issue
        full_image_path = os.path.join(base_dir, relative_path)

        if os.path.exists(full_image_path):
            shutil.move(full_image_path, os.path.join(target_dir, os.path.basename(full_image_path)))
        else:
            print(f"Warning: {full_image_path} not found!")  # Debugging

# Move images into their respective folders
move_files(train_images, train_dir)
move_files(test_images, test_dir)
move_files(val_images, val_dir)

print(f"✅ Split complete! Train: {len(train_images)}, Test: {len(test_images)}, Validation: {len(val_images)}")
