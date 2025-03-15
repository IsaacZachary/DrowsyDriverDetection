import os 
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import get_data_splits

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define dataset and model paths
dataset_path = "C:/Users/iZach/Documents/Zachh/DrowsyDriverDetection/dataset/classification_frames"
model_path = "../models/drowsy_driver_model.h5"

# Load dataset (only test set)
logging.info("Loading test dataset...")
_, _, X_test, _, _, y_test = get_data_splits(dataset_path, test_size=0.2, val_size=0.1)

# Ensure labels are correctly formatted for evaluation
y_test = y_test.squeeze()
if len(y_test.shape) == 1:
    from tensorflow.keras.utils import to_categorical
    y_test = to_categorical(y_test, 3)

# Check if model exists
if not os.path.exists(model_path):
    logging.error(f"Model file not found at {model_path}. Train the model first.")
    raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first.")

# Load the trained model
logging.info("Loading trained model...")
model = load_model(model_path)
logging.info("Model loaded successfully! ✅")

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
logging.info(f"Test Loss: {loss:.4f}")
logging.info(f"Test Accuracy: {accuracy:.4f}")

# Generate predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Print classification report
logging.info("\nClassification Report:\n" + classification_report(y_true_classes, y_pred_classes, target_names=["Alert", "Microsleep", "Yawning"]))

# Confusion matrix
logging.info("\nConfusion Matrix:\n" + str(confusion_matrix(y_true_classes, y_pred_classes)))
