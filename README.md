# 🍫 Cocoa Disease Detection Web Application

An end-to-end computer vision application designed to diagnose cocoa plant diseases and assess pod health using deep learning. Built with **PyTorch (EfficientNet-B0)** and **Streamlit**, this application provides rapid visual decision support for cocoa farmers, agronomists, and researchers.

---

## 📌 Project Overview

Cocoa (*Theobroma cacao*) is one of the world's most critical cash crops, yet annual yields are severely curtailed by fungal rot, viral infections, and insect pests. Early and accurate symptom identification is essential to preventing catastrophic crop losses.

This repository packages a pre-trained **EfficientNet-B0** image classification model into an interactive, cloud-ready web application deployable on **Streamlit Community Cloud**.

### 🌟 Key Features
- **Instant Image Classification:** Upload images (JPG, PNG, WEBP) of cocoa pods or leaves for instant inference.
- **Built-in Demonstration Samples:** Test the model immediately without needing to upload local files.
- **Detailed Probability Breakdown:** View confidence metrics and full class probability distributions powered by Altair.
- **Uncertainty Detection:** Automatic alerts when model confidence is low or ambiguous.
- **Comprehensive Pathology Guide:** Built-in clinical descriptions of all 9 recognized cocoa health categories.
- **Cloud-Ready Architecture:** Lightweight, CPU-optimized PyTorch inference designed for free-tier cloud deployment.

---

## 🔬 Model Specifications & Provenance

The underlying neural network was fine-tuned using transfer learning with an EfficientNet-B0 backbone and a customized multi-layer classification head:

- **Backbone Architecture:** EfficientNet-B0 (362 layers / parameter tensors)
- **Classification Head:** `Linear(1280 → 256) → ReLU → Dropout(0.2) → Linear(256 → 9)`
- **Input Dimensions:** `224 × 224` pixels (3 RGB channels)
- **Preprocessing Pipeline:** ImageNet standard normalization:
  - Mean: `[0.485, 0.456, 0.406]`
  - Standard Deviation: `[0.229, 0.224, 0.225]`
- **Framework:** PyTorch (`torch`, `torchvision`)
- **Model Checkpoint Size:** ~17.65 MB (`models/best_model.pth`)

### 📈 Verified Test Set Metrics
*(Extracted directly from `model_metadata.json`)*
- **Test Accuracy:** **78.48%**
- **Precision (Macro):** **80.00%**
- **Recall (Macro):** **73.15%**
- **F1 Score (Macro):** **74.67%**

---

## 🌿 Supported Disease Categories

The model is trained to recognize 9 specific classes:

| Class Key | Display Name | Pathogen / Causative Agent | Typical Field Symptoms |
|:---|:---|:---|:---|
| `anthracnose` | Anthracnose | *Colletotrichum spp.* | Necrotic leaf lesions, sunken brown pod spots |
| `black_pod_rot` | Black Pod Rot | *Phytophthora palmivora / megakarya* | Rapid dark brown/black pod rot, mummification |
| `cssvd` | Cocoa Swollen Shoot Virus | Badnavirus (vector: Mealybugs) | Red vein banding, leaf mosaic, stem swelling |
| `frosty_pod_rot`| Frosty Pod Rot | *Moniliophthora roreri* | Irregular swelling, dense white/creamy spore blanket |
| `healthy` | Healthy Pod / Leaf | N/A (Asymptomatic) | Clean, unblemished pod and foliage tissue |
| `mirid` | Mirid / Capsid Damage | *Sahlbergella singularis / Distantiella* | Dark sunken feeding punctures, shoot dieback |
| `monilia` | Moniliasis | *Moniliophthora spp.* | Destructive fungal rot affecting pod interiors |
| `pod_borer` | Cocoa Pod Borer Damage | *Conopomorpha cramerella* | Uneven ripening, larval entry/exit holes, clumping |
| `witches_broom` | Witches' Broom Disease | *Moniliophthora perniciosa* | Proliferation of broom-like shoots, cushion brooms |

### 📚 Dataset Source
The model was trained on the **Unified Cocoa Disease Dataset**:  
🔗 [Kaggle Dataset: Cocoa Disease Unified by Daniel Ohachor](https://www.kaggle.com/datasets/danielohachor/cocoa-disease-unified)

---

## 📁 Repository Structure

```text
Cocoa_Disease_prediction/
├── app.py                     # Streamlit web application entry point
├── requirements.txt           # Minimal, pinned Python dependencies
├── README.md                  # Complete project documentation
├── .gitignore                 # Excludes caches, venvs, and secrets
├── conftest.py                # Pytest path resolution configuration
├── src/                       # Application source modules
│   ├── __init__.py            # Package initialisation
│   ├── preprocessing.py       # Image validation & tensor transforms
│   └── predictor.py           # Model loading, CPU inference, and post-processing
├── models/                    # Serialized model checkpoint
│   └── best_model.pth         # PyTorch EfficientNet-B0 weights (~17.65 MB)
├── artifacts/                 # Model provenance & metadata
│   ├── class_mapping.json     # Class indices and display names
│   └── model_metadata.json    # Verified training config & metrics
├── assets/                    # Static assets & test samples
│   └── samples/               # Demo images for in-app testing
│       ├── healthy_pod_sample.jpg
│       └── diseased_pod_sample.jpg
└── tests/                     # Automated testing suite
    └── test_predictor.py      # Unit tests for preprocessing & predictor
```

---

## 💻 Local Installation & Usage

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 2. Clone Repository & Create Environment
```bash
git clone https://github.com/<your-username>/cocoa-disease-detection.git
cd cocoa-disease-detection

# Create and activate virtual environment
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run Automated Tests
```bash
python -m pytest tests/test_predictor.py -v
```

### 5. Launch the Streamlit App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ☁️ Streamlit Community Cloud Deployment Guide

Follow these steps to deploy your application to a public URL on Streamlit Community Cloud:

1. **Commit and Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Cocoa Disease Detection Streamlit App"
   git branch -M main
   git remote add origin https://github.com/<your-username>/cocoa-disease-detection.git
   git push -u origin main
   ```
2. **Access Streamlit Community Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. **Configure New App:**
   - Click **New app**.
   - **Repository:** Select your `cocoa-disease-detection` repository.
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. **Deploy:**
   - Click **Deploy!**
   - Streamlit Cloud will build the container, install packages from `requirements.txt`, load the PyTorch weights on CPU, and present your live public URL.
5. **Monitoring:**
   - Check the **Manage app** drawer at the bottom right to view build and runtime logs.

> [!NOTE]  
> Because `best_model.pth` is only **17.65 MB**, it commits directly to standard Git without requiring Git LFS or external cloud buckets, ensuring fast and reliable deployment on Streamlit Cloud.

---

## ⚠️ Agricultural Decision Support Disclaimer

> **Important Agricultural Notice:**  
> Predictions produced by this application are intended **exclusively for preliminary decision support and educational screening**. Uncalibrated softmax probabilities do not constitute an infallible diagnosis. False positives and false negatives can occur depending on lighting, camera angle, stage of infection, or symptoms resembling nutritional deficiencies.
>
> **Do not initiate chemical interventions (fungicides, insecticides, or crop destruction)** solely based on this automated prediction. Always corroborate results through on-site field evaluations conducted by a certified agronomist, agricultural extension officer, or plant pathology laboratory.
