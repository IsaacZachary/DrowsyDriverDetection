import os
import sys
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import time
import platform

# Set TensorFlow GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

# Use platform-independent sound alert
if platform.system() == "Windows":
    import winsound
    def beep(): winsound.Beep(2500, 1000)
elif platform.system() == "Darwin":
    def beep(): os.system('afplay /System/Library/Sounds/Glass.aiff')
else:
    def beep(): print("\a")  # Works in most cases, use `play -n synth 0.1 sine 1000` if needed

try:
    from data_loader import get_data_splits
except ImportError:
    logging.error("❌ Could not import `get_data_splits` from `data_loader.py`. Ensure it exists!")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define constants
IMG_SIZE = (224, 224)
NUM_CLASSES = 3
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 0.001

# Dynamically determine dataset path
dataset_path = os.path.join(os.getcwd(), "dataset", "classification_frames")
model_dir = os.path.join(os.getcwd(), "models")
os.makedirs(model_dir, exist_ok=True)  # Ensure models folder exists
model_path = os.path.join(model_dir, "drowsy_driver_model.h5")

# Load dataset with error handling
try:
    logging.info("📂 Loading dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = get_data_splits(dataset_path, test_size=0.2, val_size=0.1)
    
    # Ensure correct shape for categorical encoding
    if len(y_train.shape) == 1:
        y_train = to_categorical(y_train, NUM_CLASSES)
        y_val = to_categorical(y_val, NUM_CLASSES)
        y_test = to_categorical(y_test, NUM_CLASSES)
    
    logging.info("✅ Dataset loaded successfully!")
except Exception as e:
    logging.error(f"❌ Failed to load dataset: {e}")
    sys.exit(1)

# Define the model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Training function
def train_model():
    model = build_model()
    model.summary()
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val))
    model.save(model_path)
    logging.info(f"✅ Model training complete! Saved at: {model_path}")

# Evaluation function
def evaluate_model():
    if not os.path.exists(model_path):
        logging.error("❌ Model file not found. Train the model first.")
        return
    try:
        model = load_model(model_path)
        loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
        logging.info(f"🎯 Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
        y_pred = np.argmax(model.predict(X_test), axis=1)
        y_true = np.argmax(y_test, axis=1)
        logging.info("\n📊 Classification Report:\n" + classification_report(y_true, y_pred, target_names=["Alert", "Microsleep", "Yawning"]))
        logging.info("\n🔢 Confusion Matrix:\n" + str(confusion_matrix(y_true, y_pred)))
    except Exception as e:
        logging.error(f"❌ Error during evaluation: {e}")

# Inference (Webcam) function
def infer_from_webcam(camera_index=0):
    if not os.path.exists(model_path):
        logging.error("❌ Model file not found. Train the model first.")
        return
    try:
        model = load_model(model_path)
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            logging.error("❌ Error: Could not open webcam.")
            return
        logging.info("🎥 Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            img = cv2.resize(frame, IMG_SIZE).astype('float32') / 255.0
            img = np.expand_dims(img, axis=0)
            predictions = model.predict(img)
            predicted_class = np.argmax(predictions)
            predicted_label = ["Alert", "Microsleep", "Yawning"][predicted_class]
            confidence = np.max(predictions) * 100
            cv2.putText(frame, f"{predicted_label} ({confidence:.2f}%)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Drowsiness Detection", frame)

            if predicted_label != "Alert":
                beep()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        logging.error(f"❌ Error during webcam inference: {e}")

# Command-line argument handling
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Drowsy Driver Detection")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--infer", action="store_true", help="Run inference from webcam")
    parser.add_argument("--camera", type=int, default=0, help="Specify camera index (default: 0)")
    args = parser.parse_args()

    if args.train:
        train_model()
    elif args.evaluate:
        evaluate_model()
    elif args.infer:
        infer_from_webcam(args.camera)
    else:
        logging.error("❌ No valid argument provided. Use --train, --evaluate, or --infer")
