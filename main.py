import os
import sys

# Add scripts/src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "scripts", "src")))

from data_loader import load_data
from train import train_model
from evaluate import evaluate_model
from infer import infer

def main():
    print("Starting Drowsy Driver Detection Pipeline...")

    # Load dataset
    train_data, val_data, test_data = load_data()

    # Train model
    model = train_model(train_data, val_data)

    # Evaluate model
    evaluate_model(model, test_data)

    # Run inference on sample data
    infer(model)

if __name__ == "__main__":
    main()
