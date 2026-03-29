# SignSpeak ML Pipeline

**SignSpeak_ML_Pipeline** is a comprehensive machine learning pipeline for Pakistan Sign Language (PSL) recognition. This repository contains tools for data collection, model training, and real-time inference using MediaPipe landmark detection and LSTM neural networks.

### The Dataset is live @ the following links:
- [**Kaggle**](https://www.kaggle.com/datasets/mohib123456/dynamic-word-level-pakistan-sign-language-dataset/data)
- [**HuggingFace**](https://huggingface.co/datasets/mohibkhansherwani/DynamicWordLevelPakistanSignLanguageDataset)

## Features

- **Data Collection:** GUI-based tool to easily record and label full PSL sign sequences.
- **Data Augmentation:** Automatically expand your dataset 3-5x using time warping, spatial scaling, and sequence mirroring.
- **Model Training:** Train both standard and augmented models with automated benchmarking and checkpointing.
- **Real-Time Inference:** Live sign prediction from your webcam with stability filtering to eliminate noisy predictions.

## Tech Stack

- **Language:** Python (3.9 - 3.11)
- **Framework:** TensorFlow / Keras (LSTM Networks)
- **Computer Vision:** MediaPipe (Hand Landmarking), OpenCV

## Model Architecture & Parameters

The core machine learning model uses a Deep LSTM architecture optimized for sequential time-series data:

- **Input Shape:** `(60 frames, 126 features)` - Covers a 60-frames window mapping 21 landmarks (x, y, z) per hand.
- **Layers:** 3 LSTM layers followed by Dense layers (`tanh` activation used to prevent gradient explosion).
- **Parameters:** ~250K total parameters for high efficiency.
- **Key Hyperparameters:**
  - `SEQUENCE_LENGTH`: 60 (2.0-3.0s buffer)
  - `BATCH_SIZE`: 16
  - `EPOCHS`: 200 (with Early Stopping)

## Local Setup

### 1. Configure the Environment

Create a virtual environment to manage dependencies:

```bash
cd ml-pipeline
python -m venv venv
```

Activate the virtual environment:
- **Windows:** `.\venv\Scripts\Activate.ps1`
- **Mac/Linux:** `source venv/bin/activate`

### 2. Install Requirements

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Quick Running Guide

All scripts should be executed from the root of the ml-pipeline module.

### 1. Data Collection

Use the GUI to record and capture hand landmarks from your webcam:
```bash
python ml_pipeline_data_collection/collect_data_gui.py
```

### 2. Model Training

Train an baseline model or an augmented model (which uses mirroring/warping for better generalization):

```bash
#Training both baseline and augmented
python ml_pipeline_data_collection/train_combined.py

# Test and evauluate models
python ml_pipeline_data_collection/evaluate_models.py
```

### 3. Real-Time Inference

Test the predictions directly from your webcam:

```bash
python ml_pipeline_data_collection/realtime_inference_minimal.py
```
