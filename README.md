# Drowsy Driver Detection

## 🚀 Project Overview
The **Drowsy Driver Detection** system is an AI-powered application designed to monitor driver alertness in real-time using deep learning and computer vision. The system classifies driver states into three categories: **Alert, Microsleep, and Yawning**, helping prevent accidents caused by drowsy driving.

This project leverages **TensorFlow, OpenCV, and a Convolutional Neural Network (CNN)** model trained on a dataset of driver facial states to predict drowsiness in real-time using webcam input.

---

## Dataset
The model is trained using the **Frame Level Driver Drowsiness Detection (FL3D)** dataset from Kaggle:
- **Dataset Source:** [FL3D Dataset](https://www.kaggle.com/datasets/matjazmuc/frame-level-driver-drowsiness-detection-fl3d)
- The dataset consists of **53,331 images** labeled as:
  - **Alert**
  - **Microsleep**
  - **Yawning**

---

## 🎯 Objectives
- **Enhance Road Safety**: Reduce the risk of accidents by detecting drowsy drivers.
- **Real-time Monitoring**: Provide real-time inference using webcam feeds.
- **Accurate Classification**: Classify driver states with high precision.
- **User-Friendly Deployment**: Ensure the system is easy to set up and run.

---

## 📁 Project Structure
```
DrowsyDriverDetection/
│── dataset/                     # Dataset directory (FL3D images)
│── models/                      # Trained model directory
│   ├── drowsy_driver_model.h5   # Saved CNN model
│── scripts/
│   ├── src/
│   │   ├── train.py             # Model training script
│   │   ├── evaluate.py          # Model evaluation script
│   │   ├── infer.py             # Real-time inference script
│   │   ├── data_loader.py       # Data preprocessing and loading
│── requirements.txt             # Required dependencies
│── README.md                    # Project documentation
```

---

## 🛠️ Setup & Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/IsaacZachary/DrowsyDriverDetection.git
cd DrowsyDriverDetection
```

### 2️⃣ Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🏋️‍♂️ Model Training
To train the model on your dataset, run:
```bash
python scripts/src/train.py
```
This will preprocess the data, train the CNN model, and save it in the `models/` directory.

---

## 📊 Model Evaluation
To evaluate the trained model:
```bash
python scripts/src/evaluate.py
```
This will output the test accuracy, loss, classification report, and confusion matrix.

---

## 🎥 Real-Time Drowsiness Detection
To run real-time inference using a webcam:
```bash
python scripts/src/infer.py
```
Press **'q'** to exit the webcam feed.

---

## 🔍 Key Features
✅ **Deep Learning Model** - CNN-based classification of driver states.
✅ **Real-Time Detection** - Uses OpenCV to process live webcam input.
✅ **High Accuracy** - Trained on a diverse dataset for reliable detection.
✅ **Scalable** - Can be integrated into vehicle monitoring systems.

---

## 📝 Future Improvements
- 🚀 **Edge Deployment**: Optimize for Raspberry Pi and embedded systems.
- 🎯 **Mobile App Integration**: Develop an Android/iOS app for real-time alerts.
- 📈 **Dataset Expansion**: Train on more diverse driver images for better generalization.

---

## 🤝 Contributing
We welcome contributions! Feel free to fork the repository, create a new branch, and submit a pull request.

---

## 📩 Contact & Support
👨‍💻 **Developer:** Isaac Zachary  
📧 **Email:** [isaaczachary18@gmail.com](mailto:isaaczachary18@gmail.com)  
🔗 **GitHub:** [github.com/IsaacZachary](https://github.com/IsaacZachary)  

---

### 📜 License
This project is open-source and available under the **MIT License**.

