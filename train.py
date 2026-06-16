"""
train.py — BanglaLekha-Isolated CNN Training Script
Memory-safe training with MLflow tracking and 3-way split (train/val/test).

According to assignment requirements:
  - Splitting data into train and validation/test sets.
  - Logging parameters, metrics, and artifacts to MLflow.
  - Logging at least two training runs (runs log parameters, metrics, and artifacts).
  - Saving the best model to models/model.keras.
  - Saving class label mapping dynamically as labels.json.
"""

import os
import json
import shutil
import random
import argparse
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

# ── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE      = 64
NUM_CLASSES   = 84
SEED          = 42
MLFLOW_URI    = "./artifacts/mlflow"
MAX_PER_CLASS = 300          # Memory safe cap: 300-500 images per class

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
# STEP 1 — Build and Partition a Capped Subset Directory
# Copies max_per_class images, partitions into train/val/test splits,
# and outputs the found classes numerically sorted.
# ─────────────────────────────────────────────────────────────────────────────
def build_partitioned_subset(src_dir: str, dest_dir: str, max_per_class: int, val_split: float = 0.10, test_split: float = 0.10) -> list:
    """
    Copy up to max_per_class images per class from src_dir (folders 1..84)
    into dest_dir/train, dest_dir/val, dest_dir/test using 0-based folder naming.
    Returns the list of found class indices (as string numbers).
    """
    src = Path(src_dir)
    dst = Path(dest_dir)
    
    # Recreate clean target directory structures
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    
    for split in ["train", "val", "test"]:
        (dst / split).mkdir(parents=True, exist_ok=True)

    classes_found = []
    
    for folder_idx in range(1, NUM_CLASSES + 1):
        src_folder = src / str(folder_idx)
        if not src_folder.exists():
            continue

        class_idx = folder_idx - 1   # 0-indexed for Keras matching
        class_name = str(class_idx)
        classes_found.append(class_name)

        # Create destination directories for class_idx in each split
        for split in ["train", "val", "test"]:
            (dst / split / class_name).mkdir(parents=True, exist_ok=True)

        all_imgs = sorted(src_folder.glob("*.png"))
        selected = all_imgs[:max_per_class]

        # Deterministic shuffle for splitting reproducibility
        random.Random(SEED).shuffle(selected)

        n_total = len(selected)
        n_test = int(n_total * test_split)
        n_val = int(n_total * val_split)
        n_train = n_total - n_val - n_test

        train_imgs = selected[:n_train]
        val_imgs = selected[n_train:n_train + n_val]
        test_imgs = selected[n_train + n_val:]

        for img_path in train_imgs:
            shutil.copy2(img_path, dst / "train" / class_name / img_path.name)
        for img_path in val_imgs:
            shutil.copy2(img_path, dst / "val" / class_name / img_path.name)
        for img_path in test_imgs:
            shutil.copy2(img_path, dst / "test" / class_name / img_path.name)

        char = BANGLA_CHARS[class_idx] if class_idx < len(BANGLA_CHARS) else "?"
        print(f"  class {class_idx:>2} ({char}): train={len(train_imgs):>3}, val={len(val_imgs):>2}, test={len(test_imgs):>2}")

    # Sort numerically so string sequence matches Keras label index mapping
    classes_found.sort(key=int)
    return classes_found


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Create Datasets with Explicit class_names to Prevent Alphabetical Sorting
# ─────────────────────────────────────────────────────────────────────────────
def make_datasets(data_dir: str, class_names: list, batch_size: int):
    """
    Returns (train_ds, val_ds, test_ds) using image_dataset_from_directory.
    Passes class_names explicitly so that Keras index matches folder indices exactly.
    """
    common = dict(
        image_size       = (IMG_SIZE, IMG_SIZE),
        color_mode       = "grayscale",
        batch_size       = batch_size,
        label_mode       = "int",
        class_names      = class_names,  # Critically important to prevent scrambled alphanumeric sorting
    )

    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        directory=os.path.join(data_dir, "train"), **common
    )
    val_ds_raw = tf.keras.utils.image_dataset_from_directory(
        directory=os.path.join(data_dir, "val"), **common
    )
    test_ds_raw = tf.keras.utils.image_dataset_from_directory(
        directory=os.path.join(data_dir, "test"), **common
    )

    # Scale and invert images to [0, 1] (white text on black background)
    # This matches the zero-padding of Conv2D and prevents edge artifacts.
    train_ds = train_ds_raw.map(lambda x, y: (1.0 - (x / 255.0), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds   = val_ds_raw.map(lambda x, y: (1.0 - (x / 255.0), y),   num_parallel_calls=tf.data.AUTOTUNE)
    test_ds  = test_ds_raw.map(lambda x, y: (1.0 - (x / 255.0), y),  num_parallel_calls=tf.data.AUTOTUNE)

    # Optimize pipeline latency
    train_ds = train_ds.cache().shuffle(500, seed=SEED).prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    test_ds  = test_ds.cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds


# ─────────────────────────────────────────────────────────────────────────────
# MODEL — Lightweight CNN
# ─────────────────────────────────────────────────────────────────────────────
def build_model_v1(num_classes: int, dropout: float = 0.3) -> keras.Model:
    """
    Lightweight 3-block CNN designed to prevent OOM errors on standard hardware.
    Architecture: Input -> 3x [Conv2D -> BN -> MaxPool] -> GAP -> Dense -> Dropout -> Softmax
    """
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="input")

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="BanglaOCR_LightCNN")


def build_model_v2(num_classes: int) -> keras.Model:
    """
    Deep CNN matching the BanglaOCR_v2 architecture described in the README.
    Architecture:
      Input(64x64x1)
      -> Conv32 -> Conv32 -> BN -> MaxPool -> Dropout(0.2)
      -> Conv64 -> Conv64 -> BN -> MaxPool -> Dropout(0.2)
      -> Conv128 -> Conv256 -> BN -> MaxPool -> Dropout(0.3)
      -> GAP
      -> Dense512 -> Dropout(0.4)
      -> Dense256 -> Dropout(0.4)
      -> Dense(num_classes, softmax)
    """
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="input")

    # Block 1
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.2)(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.2)(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.3)(x)

    # Classification Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="BanglaOCR_DeepCNN")


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING — Run Experiment
# ─────────────────────────────────────────────────────────────────────────────
def run_experiment(
    run_name:   str,
    model:      keras.Model,
    train_ds,
    val_ds,
    test_ds,
    params:     dict,
    save_path:  str | None = None,
):
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("BanglaOCR")

    with mlflow.start_run(run_name=run_name):
        # Log parameters
        for k, v in params.items():
            mlflow.log_param(k, v)

        lr = params.get("learning_rate", 1e-3)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
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

        epochs = params.get("epochs", 15)
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )

        # Log epoch-wise training metrics
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

        # Evaluate on the separate test split
        test_loss, test_acc = model.evaluate(test_ds, verbose=0)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)
        print(f"[{run_name}] Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc*100:.2f}%")

        # Generate and save training curve figures
        os.makedirs("artifacts", exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history.history["accuracy"],     label="Train")
        axes[0].plot(history.history["val_accuracy"], label="Val")
        axes[0].set_title("Accuracy"); axes[0].legend()
        axes[1].plot(history.history["loss"],     label="Train")
        axes[1].plot(history.history["val_loss"], label="Val")
        axes[1].set_title("Loss"); axes[1].legend()
        curve_path = f"artifacts/{run_name}_curves.png"
        plt.savefig(curve_path, dpi=120, bbox_inches="tight")
        plt.close()
        
        mlflow.log_artifact(curve_path)

        # Log model to MLflow (with fallback)
        try:
            mlflow.keras.log_model(model, artifact_path="model")
        except Exception as e:
            print(f"[WARN] MLflow model logging failed (saving file instead): {e}")
            if os.path.exists(ckpt_path):
                mlflow.log_artifact(ckpt_path, artifact_path="model_weights")

        if save_path:
            # Overwrite/save to canonical location
            model.save(save_path)
            print(f"[INFO] Model saved → {save_path}")

    return test_acc


# ─────────────────────────────────────────────────────────────────────────────
# LABELS GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def save_labels(class_names: list, path: str = "labels.json"):
    """
    Saves index mapping mapping class index 0, 1, 2... to the actual
    Bangla characters from the original dataset labels.
    """
    mapping = {str(i): BANGLA_CHARS[int(name)] for i, name in enumerate(class_names)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"[INFO] labels.json saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="BanglaLekha OCR — Memory-Safe Partitioned Training")
    parser.add_argument(
        "--data_dir", type=str,
        default="./BanglaLekha-Isolated/Images",
        help="Path to BanglaLekha-Isolated Images folder (subfolders 1..84)"
    )
    parser.add_argument("--epochs",        type=int,   default=15)
    parser.add_argument("--batch_size",    type=int,   default=16)
    parser.add_argument("--max_per_class", type=int,   default=MAX_PER_CLASS)
    parser.add_argument("--dropout",       type=float, default=0.3)
    parser.add_argument("--lr",            type=float, default=1e-3)
    args = parser.parse_args()

    # Safety clamps
    if args.batch_size > 32:
        print("[WARN] batch_size exceeds 32. Clamping to 32 for hardware safety.")
        args.batch_size = 32
    if args.max_per_class > 500:
        print("[WARN] max_per_class exceeds 500. Clamping to 500.")
        args.max_per_class = 500

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    # 1. Verify data path exists
    data_path = Path(args.data_dir).resolve()
    if not data_path.exists():
        print(f"\n[ERROR] Data directory not found: {data_path}")
        print("[ERROR] Please provide the correct path using: python train.py --data_dir <path>")
        raise SystemExit(1)
    print(f"[INFO] Data directory verified: {data_path}")

    # 2. Build partitioned train/val/test directories
    subset_dir = "./data_subset_temp"
    print(f"\n[INFO] Partitioning subset (max {args.max_per_class}/class) → {subset_dir}")
    class_names = build_partitioned_subset(
        src_dir=str(data_path),
        dest_dir=subset_dir,
        max_per_class=args.max_per_class,
        val_split=0.10,
        test_split=0.10
    )
    print(f"[INFO] {len(class_names)} classes located.\n")

    if len(class_names) == 0:
        print("[ERROR] No valid class directories (1..84) found in the data directory.")
        shutil.rmtree(subset_dir, ignore_errors=True)
        raise SystemExit(1)

    # 3. Create datasets from subdirectories with numerically sorted classes list
    train_ds, val_ds, test_ds = make_datasets(subset_dir, class_names, args.batch_size)

    # 4. Save labels.json dynamically mapped
    save_labels(class_names, "labels.json")
    os.makedirs("artifacts/mlflow", exist_ok=True)
    os.makedirs("screenshots",      exist_ok=True)

    # 5. EXPERIMENT 1 — Standard Learning Rate (1e-3)
    print("\n" + "="*70)
    print("EXPERIMENT 1 — BanglaOCR_LightCNN (LR = 1e-3)")
    print("="*70)
    params_v1 = {
        "model":          "BanglaOCR_v1_LightCNN",
        "img_size":       IMG_SIZE,
        "num_classes":    len(class_names),
        "epochs":         args.epochs,
        "batch_size":     args.batch_size,
        "learning_rate":  args.lr,
        "dropout":        args.dropout,
        "optimizer":      "Adam",
        "architecture":   "3-block CNN + GAP",
        "max_per_class":  args.max_per_class,
    }
    model_v1 = build_model_v1(len(class_names), args.dropout)
    acc_v1 = run_experiment(
        run_name  = "run_v1_LightCNN",
        model     = model_v1,
        train_ds  = train_ds,
        val_ds    = val_ds,
        test_ds   = test_ds,
        params    = params_v1,
        save_path = "models/model.keras",
    )

    # 6. EXPERIMENT 2 — Deep CNN (v2)
    print("\n" + "="*70)
    print("EXPERIMENT 2 — BanglaOCR_v2_DeepCNN (LR = 5e-4)")
    print("="*70)
    params_v2 = {
        "model":          "BanglaOCR_v2_DeepCNN",
        "img_size":       IMG_SIZE,
        "num_classes":    len(class_names),
        "epochs":         args.epochs,
        "batch_size":     args.batch_size,
        "learning_rate":  5e-4,
        "dropout_block":  0.2,
        "dropout_head":   0.4,
        "optimizer":      "Adam",
        "architecture":   "Deep CNN (2xConv32 -> 2xConv64 -> Conv128+Conv256)",
        "max_per_class":  args.max_per_class,
    }
    model_v2 = build_model_v2(len(class_names))
    acc_v2 = run_experiment(
        run_name  = "run_v2_DeepCNN",
        model     = model_v2,
        train_ds  = train_ds,
        val_ds    = val_ds,
        test_ds   = test_ds,
        params    = params_v2,
        save_path = None,
    )

    # 7. Model Selection: Select winner based on test accuracy
    if acc_v2 > acc_v1:
        model_v2.save("models/model.keras")
        print(f"\n[INFO] Experiment 2 model outperformed Experiment 1 ({acc_v2*100:.2f}% vs {acc_v1*100:.2f}%). Saved Experiment 2 as model.keras.")
    else:
        print(f"\n[INFO] Experiment 1 model outperformed Experiment 2 ({acc_v1*100:.2f}% vs {acc_v2*100:.2f}%). Saved Experiment 1 as model.keras.")

    # 8. Clean temporary partition folders
    shutil.rmtree(subset_dir, ignore_errors=True)
    print(f"[INFO] Cleared temporary workspace files: {subset_dir}")

    print("\n✅ Training and evaluation successfully completed!")
    print("   Model Weights     → models/model.keras")
    print("   Labels Mapping    → labels.json")
    print(f"   MLflow Tracking   → {MLFLOW_URI}")


if __name__ == "__main__":
    main()
