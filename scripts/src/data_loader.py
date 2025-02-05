import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

IMG_SIZE = (224, 224)  # Target image size
LABELS = {"alert": 0, "microsleep": 1, "yawning": 2}

def load_images(dataset_path):
    """
    Load images from dataset path and preprocess them.
    
    Args:
        dataset_path (str): Path to dataset folder.

    Returns:
        np.array: Preprocessed image data.
        np.array: One-hot encoded labels.
    """
    images, labels = [], []

    for label, index in LABELS.items():
        label_path = os.path.join(dataset_path, label)
        if not os.path.exists(label_path):
            print(f"[WARNING] Skipping missing label directory: {label_path}")
            continue

        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)

            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"[WARNING] Skipping unreadable image: {img_path}")
                    continue
                
                img = cv2.resize(img, IMG_SIZE)
                img = img / 255.0  # Normalize

                images.append(img)
                labels.append(index)

            except Exception as e:
                print(f"[ERROR] Failed to process {img_path}: {e}")

    if not images:
        raise ValueError("[ERROR] No images were loaded. Check dataset path and structure.")

    return np.array(images), to_categorical(np.array(labels), num_classes=len(LABELS))

def get_data_splits(dataset_path, test_size=0.2, val_size=0.1):
    """
    Load dataset and split it into training, validation, and test sets.

    Args:
        dataset_path (str): Path to dataset folder.
        test_size (float): Proportion of data for testing.
        val_size (float): Proportion of training data for validation.

    Returns:
        Tuple of numpy arrays: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X, y = load_images(dataset_path)

    # Split into training + validation and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Further split training into training and validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    dataset_path = "dataset/FL3D"  # Update with actual dataset path
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = get_data_splits(dataset_path)
        print(f"[INFO] Train samples: {X_train.shape}")
        print(f"[INFO] Validation samples: {X_val.shape}")
        print(f"[INFO] Test samples: {X_test.shape}")
    except Exception as e:
        print(f"[ERROR] {e}")
