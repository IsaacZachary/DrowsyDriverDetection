import os
import cv2
import logging
import numpy as np
import tensorflow as tf
import time
import winsound  # For Windows beep sound (Replace with other audio for Linux/Mac)
from tensorflow.keras.models import load_model

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load trained model
model_path = "../models/drowsy_driver_model.h5"
if not os.path.exists(model_path):
    logging.error(f"Model file not found at {model_path}. Train the model first.")
    raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first.")

model = load_model(model_path)
logging.info("Model loaded successfully! ✅")

# Class labels
CLASS_LABELS = ["Alert", "Microsleep", "Yawning"]

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Better compatibility on Windows
cap.set(cv2.CAP_PROP_FPS, 30)  # Set FPS limit

if not cap.isOpened():
    logging.error("Error: Could not open webcam.")
    exit()

logging.info("Press 'q' to quit.")

def play_alert():
    """Play an alert sound if drowsiness is detected."""
    frequency = 2500  # Hz
    duration = 1000  # milliseconds
    winsound.Beep(frequency, duration)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to grab frame. Retrying...")
            time.sleep(0.1)
            continue

        # Preprocess frame
        img = cv2.resize(frame, (224, 224))  # Resize to match model input
        img = img / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)  # Expand dims for batch processing

        # Make prediction
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions)
        predicted_label = CLASS_LABELS[predicted_class]
        confidence = np.max(predictions) * 100

        # Display results
        text = f"{predicted_label} ({confidence:.2f}%)"
        color = (0, 255, 0) if predicted_label == "Alert" else (0, 0, 255)
        cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.imshow("Drowsiness Detection", frame)

        # Play alert sound if drowsiness detected
        if predicted_label in ["Microsleep", "Yawning"]:
            play_alert()

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("Exiting...")
            break

except KeyboardInterrupt:
    logging.info("Interrupted by user. Exiting...")

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    logging.info("Resources released. Program terminated.")