"""
train.py — BanglaLekha-Isolated CNN Training Script
Memory-safe training for Intel i3 / 8 GB RAM / No GPU.

Constraints (agent.md):
  - batch_size = 16  (never exceed 32)
  - image_size = 64x64
  - max 300–500 images per class
  - epochs max 15 with EarlyStopping(patience=5)
  - lightweight CNN v1 only  (v2 optional if RAM stays < 3.5 GB)
  - uses image_dataset_from_directory — NOT numpy array loading

Usage:
    python train.py --data_dir ./BanglaLekha-Isolated/Images
    python train.py --data_dir ./BanglaLekha-Isolated/Images --epochs 15 --batch_size 16 --max_per_class 300
"""

import os
import json
import shutil
import random
import argparse
import tempfile
from pathlib import Path

# Suppress TF noise before import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import mlflow
import mlflow.keras
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Constants (agent.md enforced) ────────────────────────────────────────────
IMG_SIZE      = 64           # must not change
NUM_CLASSES   = 84
SEED          = 42
MLFLOW_URI    = "./artifacts/mlflow"
MAX_PER_CLASS = 300          # agent.md: 300–500 max; use 300 for safety on 8 GB

# ── Bangla character map (index 0–83 → folder 1–84) ──────────────────────────
BANGLA_CHARS = [
    "অ","আ","ই","ঈ","উ","ঊ","ঋ","এ","ঐ","ও","ঔ",
    "ক","খ","গ","ঘ","ঙ","চ","ছ","জ","ঝ","ঞ",
    "ট","ঠ","ড","ঢ","ণ","ত","থ","দ","ধ","ন",
    "প","ফ","ব","ভ","ম","য","র","ল","শ","ষ","স","হ",
    "ড়","ঢ়","য়","ৎ","ং","ঃ","ঁ",
    "্","া","ি","ী","ু","ূ","ৃ","ে","ৈ","ো","ৌ",
    "ক্ষ",
    "০","১","২","৩","৪","৫","৬","৭","৮","৯",
    "ক্","খ্","গ্","ঘ্","চ্","জ্","ট্","ড্","ত্","ন্","ব্","ম্",
]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Build a capped subset directory
# We copy at most MAX_PER_CLASS images per class into a temp folder,
# then use image_dataset_from_directory on that folder.
# This avoids loading the whole dataset into RAM at once.
# ─────────────────────────────────────────────────────────────────────────────
def build_subset_dir(src_dir: str, dest_dir: str, max_per_class: int) -> int:
    """
    Copy up to max_per_class images per class from src_dir (folders 1..84)
    into dest_dir (folders 0..83 — 0-indexed for Keras).
    Returns the number of classes found.
    """
    src = Path(src_dir)
    dst = Path(dest_dir)
    dst.mkdir(parents=True, exist_ok=True)

    classes_found = 0
    for folder_idx in range(1, NUM_CLASSES + 1):
        src_folder = src / str(folder_idx)
        if not src_folder.exists():
            print(f"  [WARN] {src_folder} not found, skipping.")
            continue

        class_idx  = folder_idx - 1   # 0-based
        dst_folder = dst / str(class_idx)
        dst_folder.mkdir(exist_ok=True)

        all_imgs = sorted(src_folder.glob("*.png"))
        selected = all_imgs[:max_per_class]   # deterministic cap — no shuffle needed

        for img_path in selected:
            shutil.copy2(img_path, dst_folder / img_path.name)

        classes_found += 1
        char = BANGLA_CHARS[class_idx] if class_idx < len(BANGLA_CHARS) else "?"
        print(f"  class {class_idx:>2} ({char}): {len(selected):>3} images copied")

    return classes_found


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Memory-safe dataset loading via image_dataset_from_directory
# ─────────────────────────────────────────────────────────────────────────────
def make_datasets(data_dir: str, batch_size: int, val_split: float = 0.15):
    """
    Returns (train_ds, val_ds, num_classes) using image_dataset_from_directory.
    Uses .cache().prefetch() to avoid re-reading files each epoch.
    """
    common = dict(
        directory        = data_dir,
        image_size       = (IMG_SIZE, IMG_SIZE),
        color_mode       = "grayscale",
        batch_size       = batch_size,
        seed             = SEED,
        validation_split = val_split,
        label_mode       = "int",
    )

    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        subset="training", **common
    )
    val_ds_raw = tf.keras.utils.image_dataset_from_directory(
        subset="validation", **common
    )

    # Read class count BEFORE mapping (class_names lives on the raw dataset object)
    num_classes = len(train_ds_raw.class_names)
    print(f"[INFO] class_names (first 5): {train_ds_raw.class_names[:5]} ...")

    # Normalize to [0,1]
    norm = layers.Rescaling(1.0 / 255)
    train_ds = train_ds_raw.map(lambda x, y: (norm(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds   = val_ds_raw.map(lambda x, y: (norm(x), y),   num_parallel_calls=tf.data.AUTOTUNE)

    # cache() keeps data in memory after first epoch; prefetch() overlaps I/O and training
    train_ds = train_ds.cache().shuffle(500, seed=SEED).prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, num_classes


# ─────────────────────────────────────────────────────────────────────────────
# MODEL — Lightweight CNN v1 (agent.md recommended)
# ─────────────────────────────────────────────────────────────────────────────
def build_model_v1(num_classes: int, dropout: float = 0.3) -> keras.Model:
    """
    Lightweight 3-block CNN — the ONLY model to run on i3 / 8 GB.
    Architecture matches agent.md exactly:
      Input(64x64x1) → Conv32→BN→Pool → Conv64→BN→Pool → Conv128→BN→Pool
      → GAP → Dense256 → Dropout → Dense(num_classes, softmax)
    Expected size: ~15–25 MB
    """
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="input")

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="BanglaOCR_v1_LightCNN")


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING — one MLflow run
# ─────────────────────────────────────────────────────────────────────────────
def run_experiment(
    run_name:   str,
    model:      keras.Model,
    train_ds,
    val_ds,
    params:     dict,
    save_path:  str | None = None,
):
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("BanglaOCR")

    with mlflow.start_run(run_name=run_name):
        for k, v in params.items():
            mlflow.log_param(k, v)

        lr = params.get("learning_rate", 1e-3)
        model.compile(
            optimizer=keras.optimizers.Adam(lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.summary()

        os.makedirs("models", exist_ok=True)
        ckpt_path = f"models/{run_name}_best.keras"
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=5,
                restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                ckpt_path, save_best_only=True,
                monitor="val_accuracy", verbose=0
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, verbose=1
            ),
        ]

        epochs    = params.get("epochs", 15)
        history   = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )

        # Log per-epoch metrics
        for epoch, (acc, val_acc, loss, val_loss) in enumerate(zip(
            history.history["accuracy"],
            history.history["val_accuracy"],
            history.history["loss"],
            history.history["val_loss"],
        )):
            mlflow.log_metric("train_accuracy", acc,      step=epoch)
            mlflow.log_metric("val_accuracy",   val_acc,  step=epoch)
            mlflow.log_metric("train_loss",     loss,     step=epoch)
            mlflow.log_metric("val_loss",       val_loss, step=epoch)

        best_val_acc = max(history.history["val_accuracy"])
        mlflow.log_metric("best_val_accuracy", best_val_acc)
        print(f"\n[{run_name}] Best Val Accuracy: {best_val_acc*100:.2f}%")

        # Training curves
        os.makedirs("artifacts", exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history.history["accuracy"],     label="Train")
        axes[0].plot(history.history["val_accuracy"], label="Val")
        axes[0].set_title("Accuracy"); axes[0].legend()
        axes[1].plot(history.history["loss"],     label="Train")
        axes[1].plot(history.history["val_loss"], label="Val")
        axes[1].set_title("Loss"); axes[1].legend()
        curve_path = f"artifacts/{run_name}_curves.png"
        plt.savefig(curve_path, dpi=120, bbox_inches="tight"); plt.close()
        mlflow.log_artifact(curve_path)

        # Log model to MLflow
        mlflow.keras.log_model(model, artifact_path="model")

        if save_path:
            model.save(save_path)
            print(f"[INFO] Model saved → {save_path}")

    return best_val_acc


# ─────────────────────────────────────────────────────────────────────────────
# LABELS
# ─────────────────────────────────────────────────────────────────────────────
def save_labels(path: str = "labels.json"):
    mapping = {str(i): ch for i, ch in enumerate(BANGLA_CHARS)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"[INFO] labels.json saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="BanglaLekha OCR — Memory-Safe Training")
    parser.add_argument(
        "--data_dir", type=str,
        # Default: one level up from bangla-ocr-assignment/
        default="../BanglaLekha-Isolated/Images",
        help="Path to BanglaLekha-Isolated Images folder (subfolders 1..84)"
    )
    parser.add_argument("--epochs",        type=int,   default=15)   # agent.md max
    parser.add_argument("--batch_size",    type=int,   default=16)   # agent.md: must be 16
    parser.add_argument("--max_per_class", type=int,   default=MAX_PER_CLASS)
    parser.add_argument("--dropout",       type=float, default=0.3)
    parser.add_argument("--lr",            type=float, default=1e-3)
    args = parser.parse_args()

    # Safety guard from agent.md
    if args.batch_size > 32:
        print("[WARN] agent.md prohibits batch_size > 32. Clamping to 32.")
        args.batch_size = 32
    if args.max_per_class > 500:
        print("[WARN] agent.md prohibits > 500 images/class. Clamping to 500.")
        args.max_per_class = 500

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    # ── 1. Verify the data path exists ───────────────────────────────────────
    data_path = Path(args.data_dir).resolve()
    if not data_path.exists():
        print(f"\n[ERROR] Data directory not found: {data_path}")
        print("[ERROR] Usage example:")
        print("  python train.py --data_dir \"D:/Course_/Project-BanglaLekha-Isolated/BanglaLekha-Isolated/Images\"")
        raise SystemExit(1)
    print(f"[INFO] Data directory: {data_path}")

    # ── 2. Build capped subset in a temp directory ────────────────────────────
    subset_dir = "./data_subset_temp"
    print(f"\n[INFO] Building capped subset (max {args.max_per_class}/class) → {subset_dir}")
    n_classes = build_subset_dir(str(data_path), subset_dir, args.max_per_class)
    print(f"[INFO] {n_classes} classes prepared.\n")

    if n_classes == 0:
        print("[ERROR] No class folders (1..84) found in the data directory.")
        print(f"[ERROR] Checked: {data_path}")
        print("[ERROR] Make sure the Images folder contains subfolders named 1, 2, ..., 84")
        shutil.rmtree(subset_dir, ignore_errors=True)
        raise SystemExit(1)

    # ── 3. Create tf.data pipelines (memory-safe) ─────────────────────────────
    print("[INFO] Creating image_dataset_from_directory pipelines...")
    train_ds, val_ds, num_cls = make_datasets(subset_dir, args.batch_size)
    print(f"[INFO] Classes detected: {num_cls}")

    # ── 4. Save labels.json ───────────────────────────────────────────────────
    save_labels("labels.json")
    os.makedirs("artifacts/mlflow", exist_ok=True)
    os.makedirs("screenshots",      exist_ok=True)

    # ── 4. EXPERIMENT 1 — Lightweight CNN (v1) — PRIMARY ─────────────────────
    print("\n" + "="*70)
    print("EXPERIMENT 1 — BanglaOCR_v1 LightCNN (primary, agent.md recommended)")
    print("="*70)
    params_v1 = {
        "model":          "BanglaOCR_v1_LightCNN",
        "img_size":       IMG_SIZE,
        "num_classes":    num_cls,
        "epochs":         args.epochs,
        "batch_size":     args.batch_size,
        "learning_rate":  args.lr,
        "dropout":        args.dropout,
        "optimizer":      "Adam",
        "architecture":   "3-block CNN + GAP",
        "max_per_class":  args.max_per_class,
    }
    model_v1   = build_model_v1(num_cls, args.dropout)
    acc_v1     = run_experiment(
        run_name  = "run_v1_LightCNN",
        model     = model_v1,
        train_ds  = train_ds,
        val_ds    = val_ds,
        params    = params_v1,
        save_path = "models/model.keras",
    )

    # ── 5. EXPERIMENT 2 — same arch, lower LR (second MLflow run) ────────────
    # agent.md requires ≥2 MLflow runs. Run a second lightweight pass with
    # a different learning rate so it fits in RAM (no deeper v2 model locally).
    print("\n" + "="*70)
    print("EXPERIMENT 2 — BanglaOCR_v1 LightCNN (lower LR variant)")
    print("="*70)
    params_v2 = {
        "model":          "BanglaOCR_v1_LR5e4",
        "img_size":       IMG_SIZE,
        "num_classes":    num_cls,
        "epochs":         args.epochs,
        "batch_size":     args.batch_size,
        "learning_rate":  5e-4,
        "dropout":        0.4,
        "optimizer":      "Adam",
        "architecture":   "3-block CNN + GAP (lower LR)",
        "max_per_class":  args.max_per_class,
    }
    model_v2 = build_model_v1(num_cls, dropout=0.4)
    acc_v2   = run_experiment(
        run_name  = "run_v2_LR5e4",
        model     = model_v2,
        train_ds  = train_ds,
        val_ds    = val_ds,
        params    = params_v2,
        save_path = None,   # decide below
    )

    # ── 6. Pick winner and save as canonical model.keras ─────────────────────
    if acc_v2 > acc_v1:
        model_v2.save("models/model.keras")
        print(f"\n[INFO] v2 was better ({acc_v2*100:.2f}% vs {acc_v1*100:.2f}%). Saved v2 as model.keras")
    else:
        print(f"\n[INFO] v1 was better ({acc_v1*100:.2f}% vs {acc_v2*100:.2f}%). model.keras already saved.")

    # ── 7. Clean up temp subset dir ───────────────────────────────────────────
    shutil.rmtree(subset_dir, ignore_errors=True)
    print(f"[INFO] Removed temp subset dir: {subset_dir}")

    print("\n✅  Training complete!")
    print("   Model   → models/model.keras")
    print("   Labels  → labels.json")
    print(f"   MLflow  → {MLFLOW_URI}")
    print(f"\n   Next: mlflow ui --backend-store-uri {MLFLOW_URI} --port 5000")
    print("   Then:  streamlit run app.py   (shut down mlflow first!)")


if __name__ == "__main__":
    main()
