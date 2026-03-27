# SignSpeak Data Collection Module

### Important: The dataset is live @ 

### Kaggle:
 https://www.kaggle.com/datasets/mohib123456/dynamic-word-level-pakistan-sign-language-dataset/data

### HuggingFace:
 https://huggingface.co/datasets/mohibkhansherwani/DynamicWordLevelPakistanSignLanguageDataset
 
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)](https://www.tensorflow.org/)


How to run backend server: python -m uvicorn api_server:app --host 0.0.0.0 --port 8000

Run this command bypass firewall for local testing: 
Firewall rule (run power shell as admin):
New-NetFirewallRule -DisplayName "SignSpeak API" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow

**SignSpeak** is a comprehensive machine learning pipeline for Pakistan Sign Language (PSL) recognition. This repository contains tools for data collection, model training, and real-time inference using MediaPipe landmark detection and LSTM neural networks.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Mobile Recording (Android via OBS)](#mobile-recording-android-via-obs)
- [Advanced Usage](#advanced-usage)
- [Model Comparison](#model-comparison)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

SignSpeak is part of a larger Final Year Project (FYP) at COMSATS University Islamabad, Abbottabad Campus. This repository specifically handles the machine learning component:

- **Data Collection**: GUI-based tool for recording PSL sign sequences
- **Feature Extraction**: MediaPipe hand landmark detection (63 per hand)
- **Model Training**: LSTM-based deep learning with optional data augmentation
- **Inference**: Real-time sign recognition using hand landmarks

**Note**: This is the ML data collection and training module. The complete SignSpeak system includes a Flutter mobile app and FastAPI backend (developed separately).

---

## ✨ Features

### Data Collection
- ✅ Modern GUI for efficient data recording
- ✅ Pause/resume functionality for long sessions
- ✅ Real-time landmark visualization
- ✅ Progress tracking across multiple signs
- ✅ Keyboard shortcuts for streamlined workflow

### Model Training
- ✅ Baseline training (standard approach)
- ✅ Advanced training with data augmentation (3-5x dataset expansion)
- ✅ Automated model comparison and evaluation
- ✅ Early stopping and learning rate scheduling
- ✅ Model checkpointing (saves best model)

### Real-Time Inference
- ✅ Webcam-based sign recognition
- ✅ Model selection (baseline vs augmented)
- ✅ **Minimalist UI Option**: Fast, non-cluttered display for real-time testing
- ✅ Live accuracy tracking
- ✅ Performance metrics (FPS, confidence scores)

### Data Augmentation
- ✅ Time warping (speed variations)
- ✅ Horizontal flipping (left/right hand swapping)
- ✅ Spatial transformations (scaling, translation, rotation)
- ✅ Gaussian noise injection
- ✅ Temporal cropping

---

## 💻 System Requirements

### Minimum Requirements
- **Operating System**: Windows 10+, Ubuntu 20.04+, or macOS 10.15+
- **Python**: 3.9, 3.10, or 3.11 (3.11 recommended)
- **RAM**: 8 GB minimum
- **Storage**: 5 GB free space
- **Camera**: Webcam for data collection and inference

### Recommended Requirements
- **Python**: 3.11.9
- **RAM**: 16 GB
- **GPU**: NVIDIA GPU with CUDA support (optional, for faster training)

### Tested Configuration
```
OS: Windows 11
Python: 3.11.9
TensorFlow: 2.15.0
NumPy: 1.26.4
OpenCV: 4.9.0.80
MediaPipe: 0.10.9
```

---

## 🚀 Installation

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd SignSpeak-DataCollection
```

### Step 2: Create Virtual Environment

**Windows (PowerShell)**:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (CMD)**:
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Linux/macOS**:
```bash
python3 -m venv venv
source venv/bin/activate
```

> **Note**: If PowerShell gives an execution policy error:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### Step 3: Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import tensorflow as tf; import cv2; import mediapipe as mp; print('✅ All dependencies installed successfully')"
```

Expected output:
```
✅ All dependencies installed successfully
```

---

## 📁 Project Structure

```
SignSpeak-DataCollection/
├── ml_pipeline_data_collection/    # Main ML workspace
│   ├── MP_Data/                    # Collected landmark sequences (gitignored)
│   ├── actions.txt                 # List of PSL signs to recognize
│   ├── actions_config.py           # Configuration parameters
│   ├── data_augmentation.py        # Augmentation algorithms
│   ├── collect_data_gui.py         # Enhanced data collection GUI
│   ├── train_model.py              # Baseline training script
│   ├── train_model_with_augmentation.py  # Advanced training with augmentation
│   ├── compare_models.py           # Automated model comparison
│   ├── realtime_inference.py       # Basic inference script
│   ├── realtime_inference_enhanced.py    # Enhanced inference with model selection
│   ├── realtime_inference_minimal.py     # Clean, fast UI for instant feedback (Use this for real-time testing)
│   ├── action_model.h5             # Trained model (after training)
│   ├── label_encoder.pkl           # Label encoder (after training)
│   ├── mediapipe_utils.py          # Centralized MediaPipe utilities (Hands-only)
│   ├── links_to_words.txt          # Reference links to PSL dictionary
├── SRS/                            # Software Requirements Specification
│   └── srs.txt
├── SDD/                            # Software Design Document
│   └── sdd.txt
├── venv/                           # Virtual environment (gitignored)
├── requirements.txt                # Pinned Python dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

---

## 🎬 Quick Start

### 1. Prepare Actions List

Create or edit `ml_pipeline_data_collection/actions.txt`:

```bash
cd ml_pipeline_data_collection
notepad actions.txt  # Windows
# OR
nano actions.txt     # Linux/macOS
```

Add PSL signs (one per line):
```
hello
thankyou
please
yes
no
```

### 2. Collect Data

```bash
python collect_data_gui.py
```

**GUI Controls**:
- Add/remove signs using buttons
- Select sign from dropdown
- Click "START COLLECTING" to begin
- Press **SPACE** to pause/resume
- Press **ESC** to stop

**Recommended**: Collect 50 sequences per sign for optimal results.

### Mobile Recording (Android via OBS)

If DroidCam setup fails, use Android IP stream + OBS Virtual Camera.

Quick path:

1. Install `IP Webcam` on Android and start stream (`http://<phone-ip>:8080/video`)
2. In OBS, add stream as `Media Source`
3. Start OBS `Virtual Camera`
4. Run `collect_data_gui.py`, click `Rescan Cameras`, select OBS camera index

Detailed setup and validation checklist:
- `ml_pipeline_data_collection/MOBILE_RECORDING_OBS.md`

### 3. Train Model

**Option A: Baseline Model** (faster, no augmentation)
```bash
python train_model.py
```

**Option B: Augmented Model** (recommended, better accuracy)
```bash
python train_model_with_augmentation.py --augment
```

**Option C: Automated Comparison** (trains both and compares)
```bash
python compare_models.py
```

### Which training script should I use?

| Feature | `train_model.py` | `train_model_with_augmentation.py` |
|---------|------------------|------------------------------------|
| **Augmentation** | ❌ No | ✅ Yes (Optional `--augment`) |
| **Regularization** | ❌ None | ✅ Dropout layers (0.2-0.3) |
| **Early Stopping** | ❌ No | ✅ Yes (Restores best weights) |
| **LR Scheduling** | ❌ No | ✅ Yes (ReduceLROnPlateau) |
| **Checkpointing**| ❌ No | ✅ Saves `best_action_model.h5` |
| **Dataset Size** | Original (1x) | Increased (3x - 5x) |
| **Best For** | Local testing | Production / Final App |

### 4. Test Model

```bash
# Test default model
python realtime_inference.py

# OR test with model selection
python realtime_inference_enhanced.py --augmented

# 🚀 RECOMMENDED: Test with Minimalist UI (Fast & Clean)
python realtime_inference_minimal.py --augmented
```

---

## 🔬 Advanced Usage

### Data Augmentation

Augmentation effectively increases your dataset by 3-5x without additional data collection:

```bash
# 3x augmentation (recommended)
python train_model_with_augmentation.py --augment --augment-multiplier 3

# 5x augmentation (for very small datasets)
python train_model_with_augmentation.py --augment --augment-multiplier 5

# Custom epochs
python train_model_with_augmentation.py --augment --epochs 150
```

### Data Augmentation: Deep Dive

Augmentation effectively increases your dataset by **3x** without additional manual recording.

#### Augmentation Parameters & Rationale

| Technique | Probability | Range / Value | Rationale |
| :--- | :---: | :--- | :--- |
| **Horizontal Flip** | 0.3 | `x = 1.0 - x` | **Hand-Agnosticism**: Automatically teaches the model left-handed versions of signs. |
| **Time Warping** | 0.3 | 0.9x — 1.1x | **Speed Robustness**: Handles naturally fast or slow signing speeds. |
| **Spatial Scaling** | 0.2 | 0.95x — 1.05x | **Distance Invariance**: Simulates being closer or further from the camera. |
| **Translation** | 0.1 | ±5% shift | **Position Invariance**: Handles users not perfectly centered in the frame. |
| **Gaussian Noise** | 0.1 | σ = 0.01 | **Sensor Noise**: Mimics shaky hand tracking or low-light sensor jitter. |
| **Temporal Crop** | 0.2 | 10% max | **Timing Robustness**: Handles signs that start or end slightly early/late. |

**Why use "Conservative" Augmentation?**
Sign language is highly precise. Aggressive augmentation (like 90° rotation) would distort the meaning of the gestures. Our pipeline uses conservative ranges to ensure the synthetic data remains linguistically valid while still providing enough variety for the AI to learn generalization.

### Configuration

Edit `ml_pipeline_data_collection/actions_config.py`:

```python
# Recording parameters
SEQUENCE_LENGTH = 30        # Frames per sequence
NUM_SEQUENCES = 50          # Sequences per sign
FRAME_WAIT_MS = 50          # Delay between frames (ms)

# Model parameters
BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 0.001

# Inference parameters
PREDICTION_THRESHOLD = 0.5  # Minimum confidence
```

### Model Architecture

LSTM-based sequential model:
```
Input: (30 frames, 126 features)
├── LSTM(64, return_sequences=True)
├── LSTM(128, return_sequences=True)
├── LSTM(64)
├── Dense(64)
├── Dense(32)
└── Dense(num_classes, softmax)

Total params: ~250K
```

**Features**: 126 values per frame
- Left hand: 21 landmarks × 3 coords = 63
- Right hand: 21 landmarks × 3 coords = 63

---

## 📊 Model Comparison

### Comparing Baseline vs Augmented

```bash
# Automated comparison (recommended)
python compare_models.py
```

**Output**:
```
MODEL COMPARISON REPORT
==================================================
Dataset:
  Baseline:  100 sequences
  Augmented: 300 sequences (3x)

Accuracy:
  Baseline  - Train: 95.5%, Test: 78.2%  (Gap: 17.3%)
  Augmented - Train: 93.8%, Test: 89.5%  (Gap: 4.3%)

🏆 RECOMMENDATION: USE AUGMENTED MODEL
   ✅ +11.3% better test accuracy
   ✅ Reduced overfitting by 13.0%
```

### Manual Testing

```bash
# Test baseline
python realtime_inference_enhanced.py --baseline
# Perform 20 predictions, mark as correct/wrong
# Note accuracy

# Test augmented
python realtime_inference_enhanced.py --augmented
# Perform same 20 predictions
# Compare accuracy
```

**Keyboard Controls During Testing**:
- **SPACE**: Mark prediction as correct ✅
- **X**: Mark prediction as wrong ❌
- **R**: Reset statistics
- **Q**: Quit

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. Import Errors
**Problem**: `ModuleNotFoundError: No module named 'tensorflow'`

**Solution**:
```bash
# Ensure virtual environment is activated
# Look for (venv) in command prompt

# Windows
.\venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install -r requirements.txt
```

#### 2. Camera Not Detected
**Problem**: "Camera error!" or black screen

**Solution**:
- Close other apps using camera (Zoom, Teams, etc.)
- Check Windows camera permissions: Settings → Privacy → Camera
- Try different camera index in code: `cv2.VideoCapture(1)`
- If using Android via OBS:
  - Start OBS Virtual Camera first
  - Then click `Rescan Cameras` in collector
  - Select OBS camera index from dropdown
  - See `ml_pipeline_data_collection/MOBILE_RECORDING_OBS.md`

#### 3. TensorFlow Warnings
**Warning**: `oneDNN custom operations are on...`

**This is normal** - It's an informational message, not an error. To suppress:
```bash
set TF_ENABLE_ONEDNN_OPTS=0  # Windows CMD
$env:TF_ENABLE_ONEDNN_OPTS=0  # PowerShell
export TF_ENABLE_ONEDNN_OPTS=0  # Linux/macOS
```

#### 4. Low Accuracy
**Problem**: Validation accuracy < 80%

**Solutions**:
- Collect more data (aim for 50+ sequences per sign)
- Use data augmentation (`--augment`)
- Ensure consistent signing across sequences
- Check lighting conditions during data collection

#### 5. Out of Memory (OOM)
**Problem**: Training crashes with OOM error

**Solutions**:
- Reduce `BATCH_SIZE` in `actions_config.py` (try 8 or 4)
- Reduce `EPOCHS` (try 100 instead of 200)
- Close other applications
- Consider using GPU if available

### GPU Acceleration (Optional)

If you have an NVIDIA GPU:

```bash
# Uninstall CPU version

 tensorflow
pip uninstall tensorflow

# Install GPU version
pip install tensorflow[and-cuda]==2.15.0
```

Verify GPU:
```python
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))
```

---

## 📚 Documentation

Additional guides available in `/brain/` artifacts:
- `project_summary.md` - Complete FYP project overview
- `gui_features_guide.md` - Detailed GUI documentation
- `augmentation_guide.md` - Data augmentation deep dive
- `testing_workflow.md` - Comprehensive testing guide
- `inference_comparison_guide.md` - Model comparison workflow

Additional in-repo workflow guide:
- `ml_pipeline_data_collection/MOBILE_RECORDING_OBS.md` - Android phone recording via OBS Virtual Camera (no DroidCam)

---

## 🤝 Contributing

This is an academic FYP project. For issues or suggestions:

1. Check existing issues
2. Create a new issue with detailed description
3. Include error messages and system information

---

**Authors**:
- AbuZar Babar (CIIT/FA22-BSE-133/ATD)
- Mohib Ullah Khan Sherwani (CIIT/FA22-BSE-125/ATD)
- M. Abdullah Umar (CIIT/FA22-BSE-126/ATD)

**Supervisor**: Dr. Rab Nawaz Jadoon

---

## 🙏 Acknowledgments

- MediaPipe team for landmark detection library
- PSL Dictionary (psl.org.pk) for reference signs
- TensorFlow/Keras community

---

## 📞 Support

For technical issues specific to this repository:
- Check [Troubleshooting](#troubleshooting) section
- Review documentation in `/brain/` artifacts
- Create an issue with detailed logs

---

## 🔬 Technical Research & Problem Solving (FYP Evaluation)

During development, several technical challenges were identified and solved to ensure production-grade performance. These findings are critical for the FYP research component:

### 1. Hand-Dominance Variance (The "Left-Hand" Problem)
*   **Problem**: Models trained on right-handed data fail for left-handed users. Manually recording 60 signs with both hands is inefficient (120 signs total).
*   **Solution**: Implemented **Symmetric Hand-Swapping Augmentation**. By programmatically mirroring the landmark X-coordinates and swapping the Left/Right hand index labels, the dataset diversity was doubled for free.
*   **Result**: The model is now "Hand-Agnostic" and works for both left and right-handed signers.

### 2. LSTM Gradient Instability
*   **Problem**: Using `ReLU` activation in deep LSTM layers led to "Exploding Gradients" and accuracy "trash" performance as the sign count increased.
*   **Solution**: Switched architecture to use **`tanh` activation** (the gated recurrent unit standard).
*   **Result**: Training loss is now significantly more stable, allowing the model to learn 60+ signs without accuracy degradation.

### 3. Real-Time Inference "Flickering"
*   **Problem**: High-speed frame processing caused the UI to flicker between words during hand transitions.
*   **Solution**: Implemented a **Stability Locking Mechanism** (Prediction History Window). A sign is only "accepted" into the sentence if it appears in at least 6 out of the last 10 predictions.
*   **Result**: Smooth, confident text output with zero accidental word entries.

### 4. Background Noise (The "Always-On" Problem)
*   **Problem**: Without a "rest" state, the model was forced to pick a sign even when the user was just sitting still.
*   **Solution**: Integrated a **"Nothing" (Background) Class** with diverse non-signing movements (adjusting glasses, drinking water, etc.).
*   **Result**: High-fidelity idle state detection; the system remains "SILENT" until a valid sign is initiated.

### 5. Temporal Window Alignment (The "Complex Sign" Problem)
*   **Problem**: Multi-phase signs like "Goodbye" often exceeded the 1-second (30-frame) window. Even for short signs, a sliding window during inference caused "Alignment Traps" where a sign was missed if it didn't start at the exact first frame of the window.
*   **Solution**: Migrated the entire pipeline to a **60-frame (2.0 second)** window. This provides significant "Temporal Padding." 
*   **Result**: The model is now **Shift-Invariant**; it can accurately recognize a sign regardless of where it starts within the 2-second buffer.

### 6. Phantom Hand Suppression (Null-State Training)
*   **Problem**: For single-handed signs, mirroring data (augmentation) accidentally transformed "invisible" hands (zero-vectors) into "visible" landmarks at the edge of the screen ($x=1.0$), creating "Phantom Hands" that confused the model.
*   **Solution**: Implemented **Conditional Mirroring Masks**. The augmentation script now detects if a hand is missing and ensures it remains as a perfect zero-vector even after spatial flipping.
*   **Result**: Force-focuses the model on the active hand while ignoring the background, significantly reducing noise for single-handed PSL signs.

### 7. Motion Blur & Landmark Dropouts (The "Fast Sign" Problem)
*   **Problem**: Fast movements like waving caused landmarks to vanish. Built-in laptop cameras with internal software delays (20 FPS cap) created excessive motion blur that MediaPipe couldn't resolve during rapid sign execution.
*   **Solution**: Unlocked full camera frame rate by reducing `FRAME_WAIT_MS` to **1ms**. Lowered detection/tracking thresholds to **0.4** and implemented a stricter **High-Stability Filter** (requiring 9/10 consistent hits).
*   **Result**: Eliminated "Landmark Flicker" during high-speed transitions and improved tracking reliability on lower-end laptop webcams.

---


## 🔄 Recent Updates (Jan 29, 2026)

### 1. Data Collection Expansion
- **New Actions Added**: 
  - `hello` (50 sequences collected)
  - `assalam-o-alaikum` (50 sequences collected)
- **UI Optimization**: Compact mode implemented in `collect_data_gui.py` to fit smaller laptop screens (600px height).
- **Shortcut Support**: Added `S` key to start collection, reducing dependency on mouse clicks.

### 2. Augmentation Strategy Optimization
- **Problem**: Initial aggressive augmentation (15° rotation, 20% warp, 50% probability) was distorting hand landmarks, causing the augmented model to underperform compared to the baseline.
- **Fix**: Implemented a **Conservative Augmentation Strategy**:
  - Disabled `spatial_rotate` (risk of flipping sign meaning).
  - Reduced probability of all techniques to 10-30%.
  - Narrowed intensity ranges (e.g., translation reduced from 10% to 5%).
  - Result: Higher quality synthetic data that complements real data instead of confusing the model.

### 4. Technical Stabilization & FYP Documentation
- **Architecture Fix**: Switched LSTMs to `tanh` for improved training stability.
- **Hand-Dominance Implementation**: Enabled `horizontal_flip` logic to support both hands.
- **V-Sidebar UI**: Replaced horizontal sentences with a vertical scrolling log to support conversations.
- **Performance Boost**: Implemented 3:1 frame skipping in inference to remove camera lag.

---

## 🔄 Recent Updates (Jan 30, 2026)

### 1. Temporal Quality Overhaul
- **60-Frame Standard**: Upgraded the entire ML pipeline from 1.0s to 2.0s windows to support complex Pakistan Sign Language gestures.
- **Temporal Migration**: Performed a full data migration to ensure all 60 signs benefit from the new temporal padding.
- **Phantom Hand Fix**: Patched the augmentation logic to correctly handle "missing hand" states during mirroring.
- **FYP Research Documentation**: Added scientific rationales for augmentation and architecture choices to the project README.

---

---

## 🔄 Recent Updates (Jan 31, 2026)

### 1. Performance & Stability Calibration
- **FPS Bottleneck Fixed**: Reduced internal loop delay in `actions_config.py` from 50ms to 1ms, increasing capture rate from ~12 FPS to the camera's theoretical maximum (~30 FPS).
- **Enhanced Detection Sensitivity**: Adjusted MediaPipe `min_detection_confidence` to 0.4 to better handle motion-blurred hands during complex signs.
- **Workflow Automation**: Fully automated `compare_models.py` by removing manual interaction, enabling continuous pipeline execution.
- **Transition Noise Suppression**: Increased stability requirements (9/10 frame consistency) and raised `PREDICTION_THRESHOLD` to 0.8 to eliminate phantom predictions during hand transitions.
- **UI Scaling Fix**: Replaced fixed 1.5x upscaling in inference with resizable `WINDOW_NORMAL` (1000x750 default) to fix "zoomed-in" appearance.

---

**Version**: 1.2.2  
**Last Updated**: Jan 31, 2026  
**Python**: 3.9+ (3.11 recommended)  
