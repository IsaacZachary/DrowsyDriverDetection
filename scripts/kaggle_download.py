import os

DATASET_PATH = "dataset/"
DATASET_NAME = "matjazmuc/frame-level-driver-drowsiness-detection-fl3d"

def download_dataset():
    print("Downloading dataset...")
    os.system(f'kaggle datasets download -d {DATASET_NAME} -p {DATASET_PATH} --unzip')

if __name__ == "__main__":
    download_dataset()
