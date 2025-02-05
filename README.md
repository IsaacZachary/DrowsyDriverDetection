# Drowsy Driver Detection System

## Overview
This project implements a **Drowsy Driver Detection System** using deep learning. The system can classify a driver's state as **Alert, Microsleep, or Yawning** in real-time using a webcam.

## Dataset
The model is trained using the **Frame Level Driver Drowsiness Detection (FL3D)** dataset from Kaggle:
- **Dataset Source:** [FL3D Dataset](https://www.kaggle.com/datasets/matjazmuc/frame-level-driver-drowsiness-detection-fl3d)
- The dataset consists of **53,331 images** labeled as:
  - **Alert**
  - **Microsleep**
  - **Yawning**

## Project Structure
```
DrowsyDriverDetection/
│-- data/              # Dataset (Downloaded & Preprocessed Data)
│-- models/            # Saved trained models
│-- src/               # Source code
│   │-- train.py       # Model training script
│   │-- evaluate.py    # Model evaluation script
│   │-- infer.py       # Real-time inference script
│   │-- data_loader.py # Data preprocessing functions
│-- README.md          # Project documentation
│-- report.pdf         # Detailed project report
```

## Installation
### 1. Clone the Repository
```bash
git clone https://github.com/your-username/DrowsyDriverDetection.git
cd DrowsyDriverDetection
```

### 2. Create & Activate Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # For Linux/macOS
venv\Scripts\activate    # For Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the Dataset
```python
import kagglehub
path = kagglehub.dataset_download("matjazmuc/frame-level-driver-drowsiness-detection-fl3d")
print("Dataset downloaded at:", path)
```

## Training the Model
```bash
python src/train.py
```
This will preprocess the dataset, train a deep learning model, and save it in `models/drowsy_driver_model.h5`.

## Evaluating the Model
```bash
python src/evaluate.py
```
This script loads the trained model and evaluates its accuracy.

## Real-Time Detection
Run the script to detect drowsiness using a webcam:
```bash
python src/infer.py
```
Press **'q'** to exit the webcam feed.

## Results
The model will display the predicted driver state on the screen with a confidence score.

## Authors
- **Isaac Zachary**

## License
This project is licensed under the MIT License.

