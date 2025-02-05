import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from data_loader import get_data_splits

# Define constants
IMG_SIZE = (224, 224, 3)
NUM_CLASSES = 3
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# Load dataset
dataset_path = "../dataset/FL3D"  # Adjust based on the extracted dataset path
X_train, X_test, y_train, y_test = get_data_splits(dataset_path)

# Define the model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=IMG_SIZE),
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

# Train the model
history = model.fit(X_train, y_train, 
                    batch_size=BATCH_SIZE, 
                    epochs=EPOCHS, 
                    validation_data=(X_test, y_test))

# Save the trained model
model.save("../models/drowsy_driver_model.h5")
print("Model training complete! ✅ Model saved at: models/drowsy_driver_model.h5")
