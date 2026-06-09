# Bangla Handwritten Word Recognition — BanglaLekha-Isolated

> **Assignment: W11-12 · SICIP @ BRAC University**  
> A complete end-to-end OCR pipeline: CNN training → MLflow tracking → Streamlit UI → Docker deployment.

---

## Table of Contents
1. [Dataset Preparation](#1-dataset-preparation)
2. [Preprocessing Pipeline](#2-preprocessing-pipeline)
3. [Model Architecture](#3-model-architecture)
4. [Training Process & Hyperparameters](#4-training-process--hyperparameters)
5. [MLflow Tracking](#5-mlflow-tracking)
6. [Word Segmentation Strategy](#6-word-segmentation-strategy)
7. [Streamlit UI](#7-streamlit-ui)
8. [Docker Build & Run](#8-docker-build--run)
9. [Known Limitations & Improvements](#9-known-limitations--improvements)

---

## 1. Dataset Preparation

**Source:** [BanglaLekha-Isolated on Mendeley Data](https://data.mendeley.com/datasets/hf6sf8zrkc/2)

The dataset contains 84 folders (numbered `1` to `84`) where each folder represents one isolated Bangla character class. Each PNG image filename encodes district, institution, gender, age, date, form serial, and class index.

### Steps

```
Project root
└── BanglaLekha-Isolated/
    └── BanglaLekha-Isolated/
        └── Images/        ← 84 class folders
            ├── 1/
            ├── 2/
            └── ...
```

1. Download the dataset from the Mendeley link above.
2. Extract so that the `Images/` folder sits at the path shown.
3. Run training pointing to that folder (see §4).

---

## 2. Preprocessing Pipeline

| Step | Detail |
|------|--------|
| Load | `tf.keras.preprocessing.image.load_img` in **grayscale** |
| Resize | All images resized to **64 × 64** pixels |
| Normalize | Pixel values divided by **255.0** → `[0.0, 1.0]` |
| Label | Zero-based class index = folder number − 1 |
| Split | 80 % train / 10 % val / 10 % test (stratified) |

**Data augmentation** (applied in-memory during training):
- Random rotation ± 10°
- Random zoom ± 10 %
- Width/height shifts ± 10 %

---

## 3. Model Architecture

Two CNN variants are trained and tracked:

### Experiment 1 — BanglaOCR_v1 (Lightweight CNN)

```
Input (64×64×1)
  → Conv2D(32, 3×3, relu) → BN → MaxPool
  → Conv2D(64, 3×3, relu) → BN → MaxPool
  → Conv2D(128, 3×3, relu) → BN → MaxPool
  → GlobalAveragePooling2D
  → Dense(256, relu) → Dropout(0.3)
  → Dense(84, softmax)
```

| Param | Value |
|-------|-------|
| Input | 64 × 64 × 1 |
| Output classes | 84 |
| Learning rate | 0.001 |
| Dropout | 0.3 |
| Optimizer | Adam |

### Experiment 2 — BanglaOCR_v2 (Deep CNN)

```
Input (64×64×1)
  → 2 × Conv2D(32) → BN → MaxPool → Dropout(0.2)
  → 2 × Conv2D(64) → BN → MaxPool → Dropout(0.2)
  → Conv2D(128) + Conv2D(256) → BN → MaxPool → Dropout(0.3)
  → GlobalAveragePooling2D
  → Dense(512, relu) → Dropout(0.4)
  → Dense(256, relu) → Dropout(0.4)
  → Dense(84, softmax)
```

| Param | Value |
|-------|-------|
| Input | 64 × 64 × 1 |
| Output classes | 84 |
| Learning rate | 0.0005 |
| Dropout | 0.4 |
| Optimizer | Adam |

The best-performing model is saved as `models/model.keras`.

---

## 4. Training Process & Hyperparameters

```bash
# Default run (30 epochs, batch size 64)
python train.py --data_dir ../BanglaLekha-Isolated/BanglaLekha-Isolated/Images

# Custom run
python train.py \
    --data_dir ../BanglaLekha-Isolated/BanglaLekha-Isolated/Images \
    --epochs 50 \
    --batch_size 128
```

| Hyperparameter | Exp 1 | Exp 2 |
|----------------|-------|-------|
| Epochs (max) | 30 | 30 |
| Batch size | 64 | 64 |
| Learning rate | 1e-3 | 5e-4 |
| Early stopping patience | 5 | 5 |
| LR reduce patience | 3 | 3 |
| Loss | Sparse categorical cross-entropy | ← same |

**Callbacks:** `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`

---

## 5. MLflow Tracking

The MLflow tracking store is kept at `./artifacts/mlflow` so it works both locally and inside Docker.

```bash
# View the MLflow UI (run in the bangla-ocr-assignment directory)
mlflow ui --backend-store-uri ./artifacts/mlflow --host 0.0.0.0 --port 5000
```

Then open **http://localhost:5000** in your browser.

Each run logs:
- **Parameters:** model name, architecture, img_size, epochs, batch_size, lr, dropout, optimizer
- **Metrics (per epoch):** `train_accuracy`, `val_accuracy`, `train_loss`, `val_loss`
- **Summary metrics:** `test_accuracy`, `test_loss`
- **Artifacts:** classification report (`.txt`), training curves (`.png`), model

At least **two runs** are tracked by default: `run_v1_LightCNN` and `run_v2_DeepCNN`.

---

## 6. Word Segmentation Strategy

The app uses **OpenCV contour detection** to segment a drawn word into individual characters:

1. Convert RGBA canvas → BGR → Grayscale
2. Binarize with **Otsu's threshold** (inverted so strokes are white)
3. Apply **morphological closing** (3×3 kernel, 2 iterations) to connect broken strokes
4. `cv2.findContours(RETR_EXTERNAL)` — find outer character blobs
5. Filter tiny noise (area < `min_area` threshold, configurable in sidebar)
6. Sort bounding boxes **left-to-right** (standard Bangla reading order)
7. Crop each character ROI → pad to square → resize to 64×64 → normalize

This approach handles ligatures approximately; true Bangla word segmentation is a research area.

---

## 7. Streamlit UI

```bash
streamlit run app.py
```

Open **http://localhost:8501**.

**Features:**
- 🖊️ **Freehand drawing canvas** (adjustable brush size & color)
- 🔍 **Recognize Word** button → segments and classifies each character
- 📊 Displays the assembled Bangla word in large Bengali script
- 📈 Per-character confidence bars
- 🔬 Top-3 predictions expandable panel
- ⚙️ Sidebar controls for brush, background, and contour sensitivity

---

## 8. Docker Build & Run

> ⚠️ **Train the model first** (run `python train.py` locally) to generate `models/model.keras` and `labels.json` before building the image.

```bash
# 1. Build the image (from bangla-ocr-assignment/)
docker build -t bangla-ocr-app:0.1 .

# 2. Run the container
docker run -p 8501:8501 bangla-ocr-app:0.1

# Open http://localhost:8501 in your browser

# Optional: persist MLflow artifacts
docker run -p 8501:8501 -v $(pwd)/artifacts:/app/artifacts bangla-ocr-app:0.1
```

---

## 9. Known Limitations & Improvements

| Limitation | Suggested Improvement |
|------------|----------------------|
| Contour-based segmentation breaks on connected Bangla matras | Use projection-profile or deep segmentation model |
| 84 isolated character classes ≠ full word vocabulary | Add sequence model (CTC-RNN / Transformer) |
| No data augmentation in current version | Add `tf.keras.layers.RandomRotation` pipeline |
| Single canvas resolution (700×300) | Allow image upload for pre-written samples |
| Docker image doesn't ship MLflow UI | Add docker-compose with separate MLflow container |
| Bengali fonts may not render on all host systems | Bundle a Noto Bengali font in the app |

---

*Built with TensorFlow · MLflow · Streamlit · OpenCV · Docker*
