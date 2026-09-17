# Automated Road Damage Classification Using Deep Learning

An AI-based computer vision project that classifies road damage images into three categories: **Pothole, Crack, and Manhole** using Deep Learning and Transfer Learning.

## 📌 Project Overview

Road damage can affect vehicle safety and road quality. Manual inspection of roads is time-consuming and requires human effort.

This project uses a **fine-tuned ResNet50 deep learning model** to automatically classify road damage from images.

The project also includes **Grad-CAM visualization** to understand which areas of the image influenced the model's prediction.

## 🎯 Objectives

- Automatically classify road damage images.
- Detect three types of road damage:
  - Pothole
  - Crack
  - Manhole
- Compare a baseline CNN with a transfer learning model.
- Handle class imbalance using class weights.
- Use data augmentation to improve model generalization.
- Visualize model decisions using Grad-CAM.
- Deploy the trained model through a Streamlit web application.

## 📂 Dataset

The project uses the **RDD2020 Road Damage Dataset**.

The available dataset contains road images with COCO-format bounding-box annotations.

### Damage Classes

| Class | Damage Type |
|---|---|
| 0 | Pothole |
| 1 | Crack |
| 2 | Manhole |

Bounding-box annotations were used to create individual road-damage crops.

### Final Dataset

| Dataset | Images |
|---|---:|
| Training | 702 |
| Validation | 173 |
| Testing | 151 |
| **Total** | **1026** |

The dataset was split at the **original-image level** so that crops from the same original image do not appear in different splits.

## 🔧 Data Preprocessing

The following preprocessing techniques were used:

- Image resizing to **224 × 224**
- Pixel normalization
- Data augmentation
- Horizontal flipping
- Random rotation
- Random brightness adjustment
- Random zoom
- Class weighting to handle class imbalance

## 🧠 Models

### 1. Baseline CNN

A custom Convolutional Neural Network was developed as the baseline model.

It contains:

- Convolution layers
- Max pooling layers
- Fully connected layers
- Dropout
- Softmax output layer

### 2. ResNet50 Transfer Learning

A pretrained **ResNet50** model with ImageNet weights was used.

The classification head was replaced with a custom 3-class classification layer.

The model was trained in two stages:

1. Feature extraction
2. Fine-tuning

## 📊 Model Performance

| Model | Test Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Baseline CNN | 58.94% | 52.20% | 58.79% |
| ResNet50 Feature Extraction | 86.75% | 83.45% | 86.74% |
| Fine-tuned ResNet50 | **86.75%** | 82.63% | **87.04%** |

The ResNet50-based approach performed substantially better than the baseline CNN on the test dataset.

Fine-tuning maintained the same overall test accuracy while slightly reducing test loss and changing class-level performance.

## 🔍 Grad-CAM

**Grad-CAM (Gradient-weighted Class Activation Mapping)** is used to visualize the regions of an image that contributed to the model's prediction.

This improves the interpretability of the deep learning model and helps understand why a particular damage class was predicted.

## 🌐 Streamlit Web Application

A Streamlit application was developed for real-time image classification.

### Application Features

- Upload road image
- Predict road damage type
- Display prediction confidence
- Display class probabilities
- Generate Grad-CAM visualization
- Provide damage-specific recommendations

### Supported Classes

- 🕳️ Pothole
- 🛣️ Crack
- 🚧 Manhole

## 📁 Project Structure

```text
Automated-Road-Damage-Classification/
│
├── 01_Dataset_Exploration.ipynb
├── 02_Data_Preprocessing.ipynb
├── 03_CNN_Model.ipynb
├── 04_Transfer_Learning.ipynb
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── dataset/
│   └── Train / Validation / Test
│
├── models/
│   └── road_damage_resnet50_final.keras
│
└── outputs/
    └── Evaluation results and Grad-CAM visualizations
