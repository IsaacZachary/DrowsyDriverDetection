import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from data_loader import preprocess_image

# Load trained model
model_path = "../models/drowsy_driver_model.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first.")

model = load_model(model_path)
print("Model loaded successfully! ✅")

# Class labels
CLASS_LABELS = ["Alert", "Microsleep", "Yawning"]

# Open webcam for real-time inference
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Preprocess frame
    img = cv2.resize(frame, (224, 224))  # Resize to match model input
    img = preprocess_image(img)  # Normalize
    img = np.expand_dims(img, axis=0)  # Expand dims for batch processing

    # Make prediction
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    predicted_label = CLASS_LABELS[predicted_class]
    confidence = np.max(predictions) * 100

    # Display results
    cv2.putText(frame, f"{predicted_label} ({confidence:.2f}%)", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Drowsiness Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
