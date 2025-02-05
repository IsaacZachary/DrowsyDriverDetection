import os
import tensorflow as tf
from data_loader import get_data_splits
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Define dataset path
dataset_path = "../dataset/FL3D"  # Adjust this path as needed

# Load dataset
X_train, X_test, y_train, y_test = get_data_splits(dataset_path)

# Load the trained model
model_path = "../models/drowsy_driver_model.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first.")

model = load_model(model_path)
print("Model loaded successfully! ✅")

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Generate predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=["Alert", "Microsleep", "Yawning"]))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_true_classes, y_pred_classes))
