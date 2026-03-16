😷 Real-Time Face Mask Detection using VGG16
📌 Project Overview

This project implements a Real-Time Face Mask Detection System using Deep Learning and Computer Vision.
The model detects whether a person is wearing a face mask or not using a pretrained VGG16 convolutional neural network with transfer learning.

The system processes webcam images, detects faces, and classifies them into two categories:

✅ Mask

❌ No Mask

Face detection is performed using OpenCV and predictions are generated using a deep learning model trained with TensorFlow / Keras.

This type of system is widely used for public safety monitoring, especially in crowded environments such as airports, hospitals, and public transportation systems.

🎯 Objectives

Detect human faces in images or video streams

Classify faces as Mask or No Mask

Implement real-time mask detection

Use transfer learning with VGG16

Build a practical computer vision surveillance system

🧠 Model Architecture

The system uses Transfer Learning with VGG16:

Input Image (224x224x3)
        ↓
VGG16 Convolutional Layers
        ↓
Feature Extraction
        ↓
Fully Connected Layer
        ↓
Sigmoid Activation
        ↓
Binary Classification
(Mask / No Mask)

Using pretrained CNN architectures like VGG16 allows faster training and high accuracy by reusing learned image features.

⚙️ Tech Stack
Technology	Purpose
Python	Programming Language
OpenCV	Face detection & image processing
TensorFlow / Keras	Deep learning model training
VGG16	Feature extraction via transfer learning
NumPy	Numerical operations
Matplotlib	Visualization
Google Colab	Model training environment
📂 Project Structure
Face-Mask-Detection/
│
├── mask_detection.ipynb
├── model/
│   └── mask_detector.h5
│
├── dataset/
│   ├── with_mask
│   └── without_mask
│
├── images/
│   └── sample_outputs
│
└── README.md
📊 Dataset

The dataset contains images belonging to two classes:

With Mask

Without Mask

Images are resized to:

224 × 224 × 3

This resolution is required for VGG16 input format.

🚀 Features

✔ Real-time face mask detection
✔ Transfer learning using VGG16
✔ Face detection using Haar Cascade
✔ Webcam-based detection
✔ Binary classification (Mask / No Mask)
✔ Works on images and live video streams

🖥️ Installation

Clone the repository:

git clone https://github.com/yourusername/face-mask-detection.git

Install dependencies:

pip install tensorflow keras opencv-python numpy matplotlib

Run the notebook:

jupyter notebook mask_detection.ipynb
📸 Output Example
[Face Detected]
Mask (0.93)
[Face Detected]
No Mask (0.87)

The system draws a bounding box around detected faces and labels them accordingly.

📈 Future Improvements

Deploy as a web application

Use MobileNetV2 / YOLO for faster detection

Deploy on edge devices (Raspberry Pi / Jetson Nano)

Improve detection for multiple faces

Add mask compliance alerts
