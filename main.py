import os
from scripts.src.data_loader import load_data
from scripts.src.train import train_model
from scripts.src.evaluate import evaluate_model
from scripts.src.infer import infer

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
