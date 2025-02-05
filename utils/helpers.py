# utils/helpers.py
import os

def create_directories():
    dirs = ["dataset/train", "dataset/test", "dataset/validation", "models", "reports", "scripts/src", "utils"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
