import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

IMG_SIZE = (224, 224)
LABELS = {"alert": 0, "microsleep": 1, "yawning": 2}

def load_images(dataset_path):
    images, labels = [], []

    for label in LABELS.keys():
        label_path = os.path.join(dataset_path, label)
        if not os.path.exists(label_path):
            continue

        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, IMG_SIZE)
            img = img / 255.0  # Normalize

            images.append(img)
            labels.append(LABELS[label])

    return np.array(images), to_categorical(np.array(labels), num_classes=3)

def get_data_splits(dataset_path, test_size=0.2):
    X, y = load_images(dataset_path)
    return train_test_split(X, y, test_size=test_size, random_state=42)

if __name__ == "__main__":
    dataset_path = "dataset/FL3D"  # Adjust based on extracted folder name
    X_train, X_test, y_train, y_test = get_data_splits(dataset_path)
    print(f"Train samples: {X_train.shape}, Test samples: {X_test.shape}")
