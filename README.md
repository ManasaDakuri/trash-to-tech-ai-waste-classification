# Trash to Tech: An AI Model for Intelligent Waste Classification

## Overview

Improper waste segregation is a major environmental challenge that leads to pollution, health risks, and inefficient recycling. Traditional waste segregation methods often depend on manual sorting, which is time-consuming and prone to human error.

This project presents an AI-powered waste classification system that automatically classifies waste images into:

- Biodegradable Waste
- Recyclable Waste
- Hazardous Waste

The system uses a Convolutional Neural Network (CNN) with Transfer Learning based on EfficientNetB0 to identify waste categories from uploaded images. In addition to classification, the system provides a confidence score and an appropriate disposal suggestion.

---

## Features

- Automated waste image classification
- Classifies waste into three categories:
  - Biodegradable
  - Recyclable
  - Hazardous
- Confidence score generation
- Disposal recommendations
- Streamlit-based user interface
- EfficientNetB0 Transfer Learning model
- Handles class imbalance using class weights
- Data augmentation for improved generalization

---

## Problem Statement

Improper waste segregation results in:

- Environmental pollution
- Reduced recycling efficiency
- Increased health risks
- Manual sorting dependency

There is a need for an intelligent system that can automatically classify waste and assist users with proper disposal methods.

---

## Dataset

The dataset was collected from Kaggle and organized into three classes:

| Class | Description |
|---------|------------|
| Biodegradable | Organic and compostable waste |
| Recyclable | Plastic, paper, metal, glass, etc. |
| Hazardous | Batteries, electronic waste, chemicals, etc. |

Dataset preprocessing included:

- Removal of corrupted images
- Removal of empty files
- Validation of image formats
- RGB conversion
- Image resizing
- Data augmentation

---

## Methodology

### 1. Dataset Cleaning

The dataset was cleaned by:

- Removing empty files
- Removing invalid image formats
- Detecting corrupted images using:
  - PIL
  - OpenCV
  - TensorFlow
  - imghdr
- Removing very small images (<50×50 pixels)

### 2. Data Splitting

The dataset was divided using stratified sampling:

- Training Set (70%)
- Validation Set (15%)
- Test Set (15%)

This maintained class balance across all datasets.

### 3. Handling Imbalanced Data

Class weights were computed using Scikit-Learn's:

```python
compute_class_weight()
```

This prevents bias toward majority classes.

### 4. Data Augmentation

Applied augmentation techniques:

- Horizontal Flip
- Vertical Flip
- Brightness Adjustment
- Contrast Variation

### 5. Feature Extraction

The project uses:

**EfficientNetB0**

Reasons for choosing EfficientNetB0:

- Pretrained on ImageNet
- High accuracy
- Computationally efficient
- Suitable for limited datasets
- Faster convergence using Transfer Learning

### 6. Classification

The model predicts one of the following categories:

- Biodegradable
- Hazardous
- Recyclable

The predicted class is selected using a Softmax layer.

---

## Model Architecture

```text
Input Image
      ↓
Image Preprocessing
      ↓
EfficientNetB0
      ↓
Global Average Pooling
      ↓
Batch Normalization
      ↓
Dense Layer (ReLU)
      ↓
Dropout Layer
      ↓
Softmax Output Layer
      ↓
Prediction
```

---

## Technologies Used

### Programming Language

- Python

### Libraries & Frameworks

- TensorFlow
- Keras
- NumPy
- Scikit-Learn
- OpenCV
- PIL
- Matplotlib
- Streamlit

### Development Environment

- Google Colab
- VS Code

---

## Results

### Model Performance

| Metric | Value |
|----------|----------|
| Accuracy | 82% |
| Precision | 83% |
| Recall | 82% |
| F1-Score | 82% |

### Training Performance

- Stage 1 Training Accuracy: ~91%
- Fine-Tuned Validation Accuracy: ~82%
- Test Accuracy: ~81%

---

## User Interface

The Streamlit application allows users to:

1. Upload a waste image
2. Predict the waste category
3. View confidence score
4. Receive disposal recommendations

### Example Output

**Prediction:** Hazardous Waste

**Confidence:** 92.15%

**Suggestion:** Dispose in hazardous waste bin

---

## Project Structure

```text
trash-to-tech-ai-waste-classification/
│
├── app.py
├── waste_classifier.keras
├── Waste_Classification_System.ipynb
├── waste_classification_system.py
├── requirements.txt
│
├── screenshots/
│   ├── home_page.png
│   ├── prediction_example.png
│
├── report/
│   └── Mini_Project_Report.pdf
│
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/trash-to-tech-ai-waste-classification.git
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Streamlit App

```bash
streamlit run app.py
```

---

## Future Enhancements

- Integration with Smart Waste Bins
- Mobile Application Deployment
- Real-Time Waste Detection
- Multi-Object Waste Detection
- IoT-Based Waste Management Systems
- Smart City Waste Monitoring Solutions

---

## References

This project was developed after studying research papers related to:

- Waste Classification
- Deep Learning
- Transfer Learning
- EfficientNet
- Waste Detection Systems

A total of 15 research papers were reviewed during the development process.

---

## Author

**Dakuri Manasa**

B.Tech – Computer Science and Engineering

BV Raju Institute of Technology

Mini Project – 2026

---

## License

This project is licensed under the MIT License.
