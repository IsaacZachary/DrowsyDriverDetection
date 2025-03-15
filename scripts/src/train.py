import os 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from data_loader import get_data_splits  # Ensure this function correctly loads data

# Define constants
IMG_SIZE = (224, 224)  # Correct shape without channels
NUM_CLASSES = 3
BATCH_SIZE = 32
EPOCHS = 3  # Further reduced epochs to speed up training
LEARNING_RATE = 0.001

# Load dataset
dataset_path = "C:/Users/iZach/Documents/Zachh/DrowsyDriverDetection/dataset/classification_frames"

# Correct unpacking of dataset splits
X_train, X_val, X_test, y_train, y_val, y_test = get_data_splits(dataset_path, test_size=0.2, val_size=0.1)

# ✅ Debug: Check shapes before encoding
print("Before encoding:")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# ✅ Ensure labels are 1D before encoding
y_train = y_train.squeeze()
y_val = y_val.squeeze()
y_test = y_test.squeeze()

# ✅ Apply one-hot encoding (only if needed)
if len(y_train.shape) == 1:
    y_train = to_categorical(y_train, NUM_CLASSES)
    y_val = to_categorical(y_val, NUM_CLASSES)
    y_test = to_categorical(y_test, NUM_CLASSES)

# ✅ Debug: Check shapes after encoding
print("\nAfter encoding:")
print(f"y_train shape: {y_train.shape}")
print(f"y_val shape: {y_val.shape}")
print(f"y_test shape: {y_test.shape}")

# Define the model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
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

# Initialize model
model = build_model()
model.summary()

# Train the model using the correct validation data
history = model.fit(X_train, y_train, 
                    batch_size=BATCH_SIZE, 
                    epochs=EPOCHS, 
                    validation_data=(X_val, y_val))

# Save the trained model
model.save("../models/drowsy_driver_model.h5")
print("✅ Model training complete! Saved at: models/drowsy_driver_model.h5")
