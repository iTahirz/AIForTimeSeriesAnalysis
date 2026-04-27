import os
import time
import json
import warnings
import logging
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    Dropout,
    Conv1D,
    BatchNormalization,
    Input
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC, MeanAbsoluteError, RootMeanSquaredError

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

warnings.filterwarnings("ignore")


# =========================================================
# PHASE 2 — FD004 CONSERVATIVE BUT MORE BALANCED TRAINING
# =========================================================

# ---------------------------------------------------------
# SYSTEM CONFIGURATION
# ---------------------------------------------------------
# IMPORTANT: keep GPU disabled on Mac.
tf.config.set_visible_devices([], "GPU")

PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"
REPORTS_DIR = os.path.join("reports", "phase_2_training")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

LOG_FILE = os.path.join(REPORTS_DIR, "phase_2_training.log")

RUL_CAP = 125.0
FAIL_THRESHOLD = 30.0
RANDOM_STATE = 42

EPOCHS = 150
BATCH_SIZE = 128
LEARNING_RATE = 5e-4
CLIPNORM = 1.0

DROPOUT_RATE = 0.20

# More focus on RUL, but still multi-task.
RUL_OUTPUT_LOSS_WEIGHT = 4.0
FAIL_OUTPUT_LOSS_WEIGHT = 0.75

# Less aggressive than the previous conservative version.
EOL_WEIGHT_ALPHA = 3.2
EOL_WEIGHT_BETA = 3.5
OVERESTIMATION_ALPHA = 1.6
OVERESTIMATION_BETA = 2.5
UNDER_ESTIMATION_WEIGHT = 1.0

VALIDATION_SAFETY_MARGIN = 10.0
DEFAULT_FAIL_PROB_THRESHOLD = 0.50


# ---------------------------------------------------------
# LOGGER
# ---------------------------------------------------------
def build_logger() -> logging.Logger:
    logger = logging.getLogger("phase2_fd004_balanced")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


logger = build_logger()


def log_header(title: str) -> None:
    logger.info("")
    logger.info("=" * 100)
    logger.info(title)
    logger.info("=" * 100)


def log_dict(title: str, values: dict) -> None:
    logger.info(title)
    for key, value in values.items():
        logger.info(f"  - {key}: {value}")


# ---------------------------------------------------------
# REPRODUCIBILITY
# ---------------------------------------------------------
def set_reproducibility(seed: int = RANDOM_STATE) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------------------------------------------------------
# CUSTOM LOSS
# ---------------------------------------------------------
def conservative_asymmetric_rul_loss(
    eol_alpha: float = EOL_WEIGHT_ALPHA,
    eol_beta: float = EOL_WEIGHT_BETA,
    over_alpha: float = OVERESTIMATION_ALPHA,
    over_beta: float = OVERESTIMATION_BETA,
    under_weight: float = UNDER_ESTIMATION_WEIGHT,
):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        error = y_pred - y_true
        squared_error = tf.square(error)

        eol_weight = 1.0 + eol_alpha * tf.exp(-eol_beta * y_true)
        over_gate = tf.sigmoid(over_beta * error)
        over_eol_weight = tf.exp(-eol_beta * y_true)

        asym_weight = under_weight + over_alpha * over_gate * over_eol_weight

        return tf.reduce_mean(eol_weight * asym_weight * squared_error)

    return loss


# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------
def require_file(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")


def load_phase1_metadata() -> dict:
    metadata_path = os.path.join(PROCESSED_DIR, "phase_1_metadata.json")

    if not os.path.exists(metadata_path):
        logger.warning("Phase 1 metadata not found. Continuing without metadata.")
        return {}

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    logger.info(f"Phase 1 metadata loaded from: {metadata_path}")
    return metadata


def load_phase1_tensors() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    log_header("Loading Phase 1 tensors")

    tensor_map = {
        "X_train": os.path.join(PROCESSED_DIR, "X_train_fd004.npy"),
        "y_train_rul": os.path.join(PROCESSED_DIR, "y_train_fd004_rul.npy"),
        "y_train_fail": os.path.join(PROCESSED_DIR, "y_train_fd004_fail.npy"),
        "X_val": os.path.join(PROCESSED_DIR, "X_val_fd004.npy"),
        "y_val_rul": os.path.join(PROCESSED_DIR, "y_val_fd004_rul.npy"),
        "y_val_fail": os.path.join(PROCESSED_DIR, "y_val_fd004_fail.npy"),
    }

    for path in tensor_map.values():
        require_file(path)

    X_train = np.load(tensor_map["X_train"]).astype(np.float32)
    y_train_rul = np.load(tensor_map["y_train_rul"]).astype(np.float32)
    y_train_fail = np.load(tensor_map["y_train_fail"]).astype(np.float32)

    X_val = np.load(tensor_map["X_val"]).astype(np.float32)
    y_val_rul = np.load(tensor_map["y_val_rul"]).astype(np.float32)
    y_val_fail = np.load(tensor_map["y_val_fail"]).astype(np.float32)

    validate_tensors(X_train, y_train_rul, y_train_fail, "train")
    validate_tensors(X_val, y_val_rul, y_val_fail, "validation")

    return X_train, y_train_rul, y_train_fail, X_val, y_val_rul, y_val_fail


def validate_tensors(X: np.ndarray, y_rul: np.ndarray, y_fail: np.ndarray, name: str) -> None:
    if X.ndim != 3:
        raise ValueError(f"[{name}] X must be 3D. Got {X.shape}")

    if y_rul.ndim != 1:
        raise ValueError(f"[{name}] y_rul must be 1D. Got {y_rul.shape}")

    if y_fail.ndim != 1:
        raise ValueError(f"[{name}] y_fail must be 1D. Got {y_fail.shape}")

    if len(X) != len(y_rul) or len(X) != len(y_fail):
        raise ValueError(f"[{name}] Length mismatch: X={len(X)}, y_rul={len(y_rul)}, y_fail={len(y_fail)}")

    if not np.isfinite(X).all():
        raise ValueError(f"[{name}] X contains NaN or Inf.")

    if not np.isfinite(y_rul).all():
        raise ValueError(f"[{name}] y_rul contains NaN or Inf.")

    if not np.isfinite(y_fail).all():
        raise ValueError(f"[{name}] y_fail contains NaN or Inf.")

    if y_rul.min() < -1e-6 or y_rul.max() > 1.0 + 1e-6:
        raise ValueError(f"[{name}] y_rul must be normalized in [0, 1]. Got min={y_rul.min()}, max={y_rul.max()}")

    unique_fail = set(np.unique(y_fail).astype(int).tolist())
    if not unique_fail.issubset({0, 1}):
        raise ValueError(f"[{name}] y_fail must be binary. Got {sorted(unique_fail)}")


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------
def build_model(input_shape: Tuple[int, int]) -> Model:
    inputs = Input(shape=input_shape, name="sequence_input")

    x = BatchNormalization(name="input_normalization")(inputs)

    x = Conv1D(64, 3, activation="relu", padding="same", name="conv_1")(x)
    x = BatchNormalization(name="conv_1_norm")(x)

    x = Conv1D(64, 3, activation="relu", padding="same", name="conv_2")(x)
    x = BatchNormalization(name="conv_2_norm")(x)

    x = Dropout(DROPOUT_RATE, name="conv_dropout")(x)

    x = LSTM(64, return_sequences=True, name="lstm_1")(x)
    x = BatchNormalization(name="lstm_1_normalization")(x)
    x = Dropout(DROPOUT_RATE, name="lstm_1_dropout")(x)

    core = LSTM(32, name="lstm_2")(x)
    core = BatchNormalization(name="core_normalization")(core)
    core = Dropout(DROPOUT_RATE, name="core_dropout")(core)

    rul_branch = Dense(
        64,
        activation="relu",
        kernel_regularizer=l2(1e-4),
        name="rul_dense_1"
    )(core)
    rul_branch = Dropout(0.15, name="rul_dropout")(rul_branch)
    rul_output = Dense(1, activation="linear", name="rul_output")(rul_branch)

    fail_branch = Dense(32, activation="relu", name="fail_dense_1")(core)
    fail_branch = Dropout(0.15, name="fail_dropout")(fail_branch)
    fail_output = Dense(1, activation="sigmoid", name="fail_output")(fail_branch)

    return Model(
        inputs=inputs,
        outputs=[rul_output, fail_output],
        name="fd004_conservative_balanced"
    )


# ---------------------------------------------------------
# WEIGHTS
# ---------------------------------------------------------
def compute_sample_weights(
    y_train_fail: np.ndarray,
    y_val_fail: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    failure_pos_rate = float(y_train_fail.mean())
    failure_pos_weight = (1.0 - failure_pos_rate) / max(failure_pos_rate, 1e-6)

    train_fail_weights = np.where(
        y_train_fail.flatten() > 0.5,
        failure_pos_weight,
        1.0
    ).astype(np.float32)

    val_fail_weights = np.where(
        y_val_fail.flatten() > 0.5,
        failure_pos_weight,
        1.0
    ).astype(np.float32)

    train_rul_weights = np.ones((len(y_train_fail),), dtype=np.float32)
    val_rul_weights = np.ones((len(y_val_fail),), dtype=np.float32)

    return train_rul_weights, train_fail_weights, val_rul_weights, val_fail_weights, failure_pos_rate, failure_pos_weight


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------
def compute_regression_diagnostics(y_true_cycles: np.ndarray, y_pred_cycles: np.ndarray, prefix: str) -> Dict[str, float]:
    error = y_pred_cycles - y_true_cycles
    abs_error = np.abs(error)

    return {
        f"{prefix}_mae_cycles": float(np.mean(abs_error)),
        f"{prefix}_rmse_cycles": float(np.sqrt(np.mean(np.square(error)))),
        f"{prefix}_bias_cycles": float(np.mean(error)),
        f"{prefix}_median_abs_error_cycles": float(np.median(abs_error)),
        f"{prefix}_overestimation_pct": float(np.mean(error > 0.0) * 100.0),
        f"{prefix}_underestimation_pct": float(np.mean(error < 0.0) * 100.0),
        f"{prefix}_within_5_cycles_pct": float(np.mean(abs_error <= 5.0) * 100.0),
        f"{prefix}_within_10_cycles_pct": float(np.mean(abs_error <= 10.0) * 100.0),
        f"{prefix}_within_15_cycles_pct": float(np.mean(abs_error <= 15.0) * 100.0),
    }


def compute_alarm_metrics(y_true: np.ndarray, y_pred: np.ndarray, score: np.ndarray = None, prefix: str = "alarm") -> Dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_tn": int(tn),
        f"{prefix}_fp": int(fp),
        f"{prefix}_fn": int(fn),
        f"{prefix}_tp": int(tp),
        f"{prefix}_confusion_matrix": cm.tolist(),
    }

    if score is not None and len(np.unique(y_true)) == 2:
        metrics[f"{prefix}_roc_auc"] = float(roc_auc_score(y_true, score))
        metrics[f"{prefix}_average_precision"] = float(average_precision_score(y_true, score))

    return metrics


def find_best_thresholds(y_true_alarm: np.ndarray, scores: np.ndarray) -> Dict[str, dict]:
    thresholds = np.linspace(0.01, 0.99, 99)
    rows = []

    best_f1 = {"threshold": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0}
    best_recall_90 = None
    best_recall_95 = None

    for threshold in thresholds:
        pred = (scores >= threshold).astype(int)

        precision = precision_score(y_true_alarm, pred, zero_division=0)
        recall = recall_score(y_true_alarm, pred, zero_division=0)
        f1 = f1_score(y_true_alarm, pred, zero_division=0)
        accuracy = accuracy_score(y_true_alarm, pred)

        row = {
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy),
        }

        rows.append(row)

        if f1 > best_f1["f1"]:
            best_f1 = row.copy()

        if recall >= 0.90:
            if best_recall_90 is None or precision > best_recall_90["precision"]:
                best_recall_90 = row.copy()

        if recall >= 0.95:
            if best_recall_95 is None or precision > best_recall_95["precision"]:
                best_recall_95 = row.copy()

    return {
        "best_f1": best_f1,
        "best_precision_with_recall_at_least_90": best_recall_90,
        "best_precision_with_recall_at_least_95": best_recall_95,
        "rows": rows,
    }


def compute_validation_diagnostics(
    y_true_rul: np.ndarray,
    y_pred_rul: np.ndarray,
    y_true_fail: np.ndarray,
    y_pred_fail_prob: np.ndarray,
    fail_threshold: float,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    y_true_cycles = y_true_rul.reshape(-1) * RUL_CAP
    y_pred_cycles = np.clip(y_pred_rul.reshape(-1) * RUL_CAP, 0.0, RUL_CAP)

    y_true_fail_flat = y_true_fail.reshape(-1).astype(int)
    y_pred_fail_prob_flat = y_pred_fail_prob.reshape(-1)

    error = y_pred_cycles - y_true_cycles

    eol_mask = y_true_cycles <= FAIL_THRESHOLD
    mid_mask = (y_true_cycles > FAIL_THRESHOLD) & (y_true_cycles <= 80.0)
    high_mask = y_true_cycles > 80.0

    safe_rul = np.clip(y_pred_cycles - VALIDATION_SAFETY_MARGIN, 0.0, RUL_CAP)

    safety_alarm = (safe_rul <= FAIL_THRESHOLD).astype(int)
    neural_alarm = (y_pred_fail_prob_flat >= fail_threshold).astype(int)
    hybrid_alarm = np.logical_or(safety_alarm == 1, neural_alarm == 1).astype(int)

    diagnostics: Dict[str, object] = {}

    diagnostics.update(compute_regression_diagnostics(y_true_cycles, y_pred_cycles, "validation_global"))
    diagnostics.update(compute_regression_diagnostics(y_true_cycles, safe_rul, "validation_safety_adjusted"))

    if np.sum(eol_mask) > 0:
        diagnostics.update(compute_regression_diagnostics(y_true_cycles[eol_mask], y_pred_cycles[eol_mask], "validation_eol"))
        diagnostics["validation_eol_samples"] = int(np.sum(eol_mask))

    if np.sum(mid_mask) > 0:
        diagnostics.update(compute_regression_diagnostics(y_true_cycles[mid_mask], y_pred_cycles[mid_mask], "validation_mid_rul"))
        diagnostics["validation_mid_rul_samples"] = int(np.sum(mid_mask))

    if np.sum(high_mask) > 0:
        diagnostics.update(compute_regression_diagnostics(y_true_cycles[high_mask], y_pred_cycles[high_mask], "validation_high_rul"))
        diagnostics["validation_high_rul_samples"] = int(np.sum(high_mask))

    diagnostics.update(compute_alarm_metrics(y_true_fail_flat, neural_alarm, y_pred_fail_prob_flat, "validation_neural_alarm"))
    diagnostics.update(compute_alarm_metrics(y_true_fail_flat, safety_alarm, -safe_rul, "validation_safety_alarm"))
    diagnostics.update(
        compute_alarm_metrics(
            y_true_fail_flat,
            hybrid_alarm,
            np.maximum(y_pred_fail_prob_flat, (RUL_CAP - safe_rul) / RUL_CAP),
            "validation_hybrid_alarm",
        )
    )

    diagnostics["selected_failure_probability_threshold"] = float(fail_threshold)
    diagnostics["validation_safety_margin_cycles"] = float(VALIDATION_SAFETY_MARGIN)

    arrays = {
        "y_true_cycles": y_true_cycles,
        "y_pred_cycles": y_pred_cycles,
        "safe_rul": safe_rul,
        "error": error,
        "abs_error": np.abs(error),
        "y_true_fail": y_true_fail_flat,
        "y_pred_fail_prob": y_pred_fail_prob_flat,
        "neural_alarm": neural_alarm,
        "safety_alarm": safety_alarm,
        "hybrid_alarm": hybrid_alarm,
    }

    return diagnostics, arrays


# ---------------------------------------------------------
# PLOTS
# ---------------------------------------------------------
def save_current_plot(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, filename))
    plt.close()
    logger.info(f"Plot saved: {os.path.join(REPORTS_DIR, filename)}")


def generate_training_plots(history) -> None:
    log_header("Generating training curves")

    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("paper", font_scale=1.1)

    epochs = range(1, len(history.history["loss"]) + 1)

    plot_specs = [
        ("loss", "val_loss", "Multi-task Loss", "Loss", "01_loss_curve.png"),
        ("rul_output_loss", "val_rul_output_loss", "Conservative RUL Loss", "Weighted MSE", "02_rul_conservative_loss.png"),
        ("rul_output_mae", "val_rul_output_mae", "RUL MAE", "Normalized MAE", "03_rul_mae.png"),
        ("rul_output_rmse", "val_rul_output_rmse", "RUL RMSE", "Normalized RMSE", "04_rul_rmse.png"),
        ("fail_output_binary_accuracy", "val_fail_output_binary_accuracy", "Failure Classification Accuracy", "Accuracy", "05_fail_accuracy.png"),
        ("fail_output_precision", "val_fail_output_precision", "Failure Classification Precision", "Precision", "06_fail_precision.png"),
        ("fail_output_recall", "val_fail_output_recall", "Failure Classification Recall", "Recall", "07_fail_recall.png"),
        ("fail_output_auc", "val_fail_output_auc", "Failure Classification AUC", "AUC", "08_fail_auc.png"),
    ]

    for train_key, val_key, title, ylabel, filename in plot_specs:
        if train_key in history.history and val_key in history.history:
            plt.figure(figsize=(10, 6), dpi=300)
            plt.plot(epochs, history.history[train_key], linewidth=2, label="Training")
            plt.plot(epochs, history.history[val_key], linestyle="--", linewidth=2, label="Validation")
            plt.title(title, fontweight="bold")
            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.legend()
            save_current_plot(filename)

    lr_key = "learning_rate" if "learning_rate" in history.history else "lr" if "lr" in history.history else None

    if lr_key is not None:
        plt.figure(figsize=(10, 6), dpi=300)
        plt.plot(epochs, history.history[lr_key], linewidth=2)
        plt.title("Learning Rate Schedule", fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Learning rate")
        save_current_plot("09_learning_rate.png")


def plot_validation_diagnostics(arrays: Dict[str, np.ndarray]) -> None:
    log_header("Generating validation diagnostics")

    y_true_cycles = arrays["y_true_cycles"]
    y_pred_cycles = arrays["y_pred_cycles"]
    safe_rul = arrays["safe_rul"]
    error = arrays["error"]
    abs_error = arrays["abs_error"]
    y_true_fail = arrays["y_true_fail"]
    y_pred_fail_prob = arrays["y_pred_fail_prob"]

    eol_mask = y_true_cycles <= FAIL_THRESHOLD

    plt.figure(figsize=(10, 8), dpi=300)
    scatter = plt.scatter(
        y_true_cycles,
        y_pred_cycles,
        c=y_pred_fail_prob,
        cmap="coolwarm",
        alpha=0.75,
        edgecolor="k",
        s=25
    )
    plt.plot([0, RUL_CAP], [0, RUL_CAP], "k--", linewidth=2, label="Ideal prediction")
    plt.fill_between([0, RUL_CAP], [-10, RUL_CAP - 10], [10, RUL_CAP + 10], alpha=0.12, label="±10 cycle band")
    plt.axvline(FAIL_THRESHOLD, linestyle=":", linewidth=2, label="Failure threshold")
    plt.colorbar(scatter, label="Predicted failure probability")
    plt.title("Validation: True RUL vs Predicted RUL", fontweight="bold")
    plt.xlabel("True RUL cycles", fontweight="bold")
    plt.ylabel("Predicted RUL cycles", fontweight="bold")
    plt.xlim(0, RUL_CAP)
    plt.ylim(0, RUL_CAP)
    plt.legend()
    save_current_plot("10_validation_rul_scatter.png")

    plt.figure(figsize=(10, 6), dpi=300)
    sns.kdeplot(error, fill=True, alpha=0.5, label="Raw error")
    sns.kdeplot(safe_rul - y_true_cycles, fill=True, alpha=0.5, label="Safety-adjusted error")
    plt.axvline(0, linestyle="--", linewidth=2, label="Zero error")
    plt.title("Validation Error Distribution", fontweight="bold")
    plt.xlabel("Predicted RUL - True RUL cycles", fontweight="bold")
    plt.ylabel("Density", fontweight="bold")
    plt.legend()
    save_current_plot("11_validation_error_distribution.png")

    if np.sum(eol_mask) > 0:
        plt.figure(figsize=(10, 6), dpi=300)
        sns.histplot(error[eol_mask], bins=30, kde=True, alpha=0.65)
        plt.axvline(0, linestyle="--", linewidth=2, label="Zero error")
        plt.title("Validation End-of-Life Error Distribution", fontweight="bold")
        plt.xlabel("EOL error cycles", fontweight="bold")
        plt.ylabel("Count", fontweight="bold")
        plt.legend()
        save_current_plot("12_validation_eol_error_distribution.png")

    plt.figure(figsize=(10, 6), dpi=300)
    plt.scatter(y_true_cycles, abs_error, alpha=0.5, s=20)
    plt.axvline(FAIL_THRESHOLD, linestyle=":", linewidth=2, label="Failure threshold")
    plt.gca().invert_xaxis()
    plt.title("Validation Absolute Error as Failure Approaches", fontweight="bold")
    plt.xlabel("True RUL cycles", fontweight="bold")
    plt.ylabel("Absolute error cycles", fontweight="bold")
    plt.legend()
    save_current_plot("13_validation_abs_error_vs_rul.png")

    plt.figure(figsize=(10, 6), dpi=300)
    sns.histplot(
        x=y_pred_fail_prob,
        hue=y_true_fail.astype(int),
        bins=30,
        stat="density",
        common_norm=False,
        alpha=0.55
    )
    plt.axvline(DEFAULT_FAIL_PROB_THRESHOLD, linestyle="--", linewidth=2, label="Default threshold")
    plt.title("Validation Failure Probability Distribution", fontweight="bold")
    plt.xlabel("Predicted failure probability", fontweight="bold")
    plt.ylabel("Density", fontweight="bold")
    plt.legend()
    save_current_plot("14_validation_failure_probability_distribution.png")

    if len(np.unique(y_true_fail)) == 2:
        fpr, tpr, _ = roc_curve(y_true_fail, y_pred_fail_prob)
        auc_value = roc_auc_score(y_true_fail, y_pred_fail_prob)

        plt.figure(figsize=(8, 6), dpi=300)
        plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc_value:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Random")
        plt.title("Validation Failure ROC Curve", fontweight="bold")
        plt.xlabel("False positive rate", fontweight="bold")
        plt.ylabel("True positive rate", fontweight="bold")
        plt.legend()
        save_current_plot("15_validation_failure_roc_curve.png")

        precision, recall, _ = precision_recall_curve(y_true_fail, y_pred_fail_prob)
        ap_value = average_precision_score(y_true_fail, y_pred_fail_prob)

        plt.figure(figsize=(8, 6), dpi=300)
        plt.plot(recall, precision, linewidth=2, label=f"AP = {ap_value:.3f}")
        plt.title("Validation Failure Precision-Recall Curve", fontweight="bold")
        plt.xlabel("Recall", fontweight="bold")
        plt.ylabel("Precision", fontweight="bold")
        plt.legend()
        save_current_plot("16_validation_failure_precision_recall_curve.png")


def plot_threshold_sweep(rows: list) -> None:
    import pandas as pd

    sweep_df = pd.DataFrame(rows)
    sweep_path = os.path.join(REPORTS_DIR, "validation_failure_threshold_sweep.csv")
    sweep_df.to_csv(sweep_path, index=False)

    logger.info(f"Threshold sweep saved: {sweep_path}")

    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(sweep_df["threshold"], sweep_df["precision"], linewidth=2, label="Precision")
    plt.plot(sweep_df["threshold"], sweep_df["recall"], linewidth=2, label="Recall")
    plt.plot(sweep_df["threshold"], sweep_df["f1"], linewidth=2, label="F1")
    plt.axvline(DEFAULT_FAIL_PROB_THRESHOLD, linestyle="--", linewidth=1.5, label="Default threshold")
    plt.title("Validation Failure Threshold Sweep", fontweight="bold")
    plt.xlabel("Threshold", fontweight="bold")
    plt.ylabel("Metric", fontweight="bold")
    plt.legend()
    save_current_plot("17_validation_threshold_sweep.png")


# ---------------------------------------------------------
# REPORTING
# ---------------------------------------------------------
def dataframe_to_markdown_table(rows: list) -> str:
    if not rows:
        return "_No data available._"

    headers = list(rows[0].keys())
    lines = []

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in rows:
        cells = []

        for header in headers:
            value = row[header]

            if isinstance(value, (float, np.floating)):
                cells.append(f"{float(value):.6f}")
            else:
                cells.append(str(value))

        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def save_json(path: str, payload: dict) -> None:
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, default=convert)

    logger.info(f"JSON saved: {path}")


def save_training_report(val_metrics: dict, diagnostics: dict, training_config: dict, threshold_selection: dict, metadata: dict) -> None:
    report_path = os.path.join(REPORTS_DIR, "phase_2_training_report.md")

    metric_rows = []
    for key, value in val_metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            metric_rows.append({"Metric": key, "Value": float(value)})

    diagnostic_rows = []
    for key, value in diagnostics.items():
        if isinstance(value, (int, float, np.integer, np.floating)) or value is None:
            diagnostic_rows.append({"Metric": key, "Value": value})

    threshold_rows = []
    for name, result in threshold_selection.items():
        if isinstance(result, dict):
            threshold_rows.append({
                "Strategy": name,
                "Threshold": result.get("threshold"),
                "Precision": result.get("precision"),
                "Recall": result.get("recall"),
                "F1": result.get("f1"),
                "Accuracy": result.get("accuracy"),
            })

    content = f"""# Phase 2 — FD004 Conservative Balanced Multi-task Training Report

## Objective
Train a CNN-LSTM multi-task prognostic model for RUL regression and failure-state classification.

This version keeps the original stable architecture and only adjusts the training objective to reduce excessive conservative bias.

## Phase 1 Input
- Window size: {metadata.get("window_size", "unknown")}
- Feature count: {metadata.get("final_feature_count", metadata.get("feature_count", "unknown"))}

## Conservative Loss
- EOL weight alpha: {EOL_WEIGHT_ALPHA}
- EOL weight beta: {EOL_WEIGHT_BETA}
- Overestimation alpha: {OVERESTIMATION_ALPHA}
- Overestimation beta: {OVERESTIMATION_BETA}
- Underestimation weight: {UNDER_ESTIMATION_WEIGHT}

## Training Configuration
- Epochs: {training_config["epochs"]}
- Batch size: {training_config["batch_size"]}
- Learning rate: {training_config["learning_rate"]}
- Gradient clipnorm: {training_config["clipnorm"]}
- Dropout rate: {training_config["dropout_rate"]}
- RUL output loss weight: {training_config["rul_output_loss_weight"]}
- Failure output loss weight: {training_config["fail_output_loss_weight"]}
- Scheduler monitor: val_rul_output_mae
- Scheduler factor: 0.85
- Scheduler patience: 8
- Early stopping monitor: val_rul_output_mae
- GPU disabled: {training_config["gpu_disabled"]}

## Final Validation Metrics

{dataframe_to_markdown_table(metric_rows)}

## Validation Diagnostics

{dataframe_to_markdown_table(diagnostic_rows)}

## Validation Threshold Selection

{dataframe_to_markdown_table(threshold_rows)}

## Notes
Keras RUL metrics are normalized. Cycle-level diagnostics multiply predictions by {RUL_CAP:.0f}.
The selected failure threshold is chosen on validation only and should be reused unchanged during official test evaluation.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"Markdown report saved: {report_path}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main() -> None:
    start_time = time.time()
    set_reproducibility(RANDOM_STATE)

    log_header("PHASE 2 - FD004 CONSERVATIVE BALANCED TRAINING")

    metadata = load_phase1_metadata()

    X_train, y_train_rul, y_train_fail, X_val, y_val_rul, y_val_fail = load_phase1_tensors()

    y_train_rul = np.expand_dims(y_train_rul, axis=-1)
    y_train_fail = np.expand_dims(y_train_fail, axis=-1)
    y_val_rul = np.expand_dims(y_val_rul, axis=-1)
    y_val_fail = np.expand_dims(y_val_fail, axis=-1)

    log_header("Dataset diagnostics")

    log_dict("Training tensor summary", {
        "X_train shape": X_train.shape,
        "y_train_rul shape": y_train_rul.shape,
        "y_train_fail shape": y_train_fail.shape,
        "RUL target min": float(y_train_rul.min()),
        "RUL target max": float(y_train_rul.max()),
        "Failure positive rate": float(y_train_fail.mean()),
    })

    log_dict("Validation tensor summary", {
        "X_val shape": X_val.shape,
        "y_val_rul shape": y_val_rul.shape,
        "y_val_fail shape": y_val_fail.shape,
        "RUL target min": float(y_val_rul.min()),
        "RUL target max": float(y_val_rul.max()),
        "Failure positive rate": float(y_val_fail.mean()),
    })

    train_rul_weights, train_fail_weights, val_rul_weights, val_fail_weights, failure_pos_rate, failure_pos_weight = compute_sample_weights(
        y_train_fail=y_train_fail,
        y_val_fail=y_val_fail,
    )

    training_config = {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "clipnorm": CLIPNORM,
        "dropout_rate": DROPOUT_RATE,
        "rul_output_loss_weight": RUL_OUTPUT_LOSS_WEIGHT,
        "fail_output_loss_weight": FAIL_OUTPUT_LOSS_WEIGHT,
        "eol_weight_alpha": EOL_WEIGHT_ALPHA,
        "eol_weight_beta": EOL_WEIGHT_BETA,
        "overestimation_alpha": OVERESTIMATION_ALPHA,
        "overestimation_beta": OVERESTIMATION_BETA,
        "underestimation_weight": UNDER_ESTIMATION_WEIGHT,
        "failure_positive_rate": failure_pos_rate,
        "failure_positive_weight": float(failure_pos_weight),
        "validation_safety_margin": VALIDATION_SAFETY_MARGIN,
        "default_fail_probability_threshold": DEFAULT_FAIL_PROB_THRESHOLD,
        "gpu_disabled": True,
        "random_state": RANDOM_STATE,
    }

    save_json(os.path.join(REPORTS_DIR, "training_configuration.json"), training_config)
    log_dict("Training configuration", training_config)

    log_header("Model construction")

    model = build_model((X_train.shape[1], X_train.shape[2]))

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        clipnorm=CLIPNORM
    )

    model.compile(
        optimizer=optimizer,
        loss={
            "rul_output": conservative_asymmetric_rul_loss(),
            "fail_output": "binary_crossentropy",
        },
        loss_weights={
            "rul_output": RUL_OUTPUT_LOSS_WEIGHT,
            "fail_output": FAIL_OUTPUT_LOSS_WEIGHT,
        },
        metrics={
            "rul_output": [
                MeanAbsoluteError(name="mae"),
                RootMeanSquaredError(name="rmse"),
            ],
            "fail_output": [
                BinaryAccuracy(name="binary_accuracy"),
                Precision(name="precision"),
                Recall(name="recall"),
                AUC(name="auc"),
            ],
        },
    )

    model.summary(print_fn=logger.info)

    callbacks = [
        EarlyStopping(
            monitor="val_rul_output_mae",
            patience=25,
            restore_best_weights=True,
            mode="min",
            verbose=1,
        ),
        ModelCheckpoint(
            os.path.join(MODELS_DIR, "fd004_specialist.keras"),
            monitor="val_rul_output_mae",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_rul_output_mae",
            factor=0.85,
            patience=8,
            mode="min",
            min_lr=5e-6,
            verbose=1,
        ),
        CSVLogger(
            os.path.join(REPORTS_DIR, "training_history.csv"),
            append=False
        ),
    ]

    log_header("Training started")

    history = model.fit(
        X_train,
        [y_train_rul, y_train_fail],
        validation_data=(
            X_val,
            [y_val_rul, y_val_fail],
            [val_rul_weights, val_fail_weights],
        ),
        sample_weight=[
            train_rul_weights,
            train_fail_weights,
        ],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    generate_training_plots(history)

    log_header("Validation evaluation")

    val_metrics = model.evaluate(
        X_val,
        [y_val_rul, y_val_fail],
        sample_weight=[
            val_rul_weights,
            val_fail_weights,
        ],
        verbose=0,
        return_dict=True,
    )

    log_dict("Weighted validation metrics", {k: f"{v:.6f}" for k, v in val_metrics.items()})

    predictions = model.predict(X_val, verbose=0)

    y_val_pred_rul = predictions[0].reshape(-1)
    y_val_pred_fail_prob = predictions[1].reshape(-1)

    threshold_selection = find_best_thresholds(
        y_val_fail.reshape(-1).astype(int),
        y_val_pred_fail_prob
    )

    selected_threshold_record = threshold_selection.get("best_precision_with_recall_at_least_95")
    selected_strategy = "best_precision_with_recall_at_least_95"

    if selected_threshold_record is None:
        selected_threshold_record = threshold_selection.get("best_precision_with_recall_at_least_90")
        selected_strategy = "best_precision_with_recall_at_least_90"

    if selected_threshold_record is None:
        selected_threshold_record = threshold_selection.get("best_f1")
        selected_strategy = "best_f1"

    selected_threshold = float(selected_threshold_record["threshold"])

    selected_threshold_payload = {
        "selected_threshold": selected_threshold,
        "selected_strategy": selected_strategy,
        "selection_source": "validation",
        "selection_reason": "Validation-selected threshold. Conservative policy prioritizes high recall while avoiding test-set tuning.",
        "default_threshold": DEFAULT_FAIL_PROB_THRESHOLD,
        "threshold_selection": {
            key: value for key, value in threshold_selection.items() if key != "rows"
        },
    }

    save_json(
        os.path.join(REPORTS_DIR, "selected_failure_threshold.json"),
        selected_threshold_payload
    )

    plot_threshold_sweep(threshold_selection["rows"])

    diagnostics, arrays = compute_validation_diagnostics(
        y_true_rul=y_val_rul,
        y_pred_rul=y_val_pred_rul,
        y_true_fail=y_val_fail,
        y_pred_fail_prob=y_val_pred_fail_prob,
        fail_threshold=selected_threshold,
    )

    log_header("Validation prognostic diagnostics")
    log_dict("Validation diagnostics", diagnostics)

    plot_validation_diagnostics(arrays)

    results = {
        "validation_metrics_weighted": {k: float(v) for k, v in val_metrics.items()},
        "validation_diagnostics": diagnostics,
        "training_configuration": training_config,
        "selected_failure_threshold": selected_threshold_payload,
        "phase1_metadata": metadata,
    }

    save_json(os.path.join(REPORTS_DIR, "validation_results.json"), results)

    np.save(os.path.join(REPORTS_DIR, "val_true_rul_cycles.npy"), arrays["y_true_cycles"])
    np.save(os.path.join(REPORTS_DIR, "val_pred_rul_cycles.npy"), arrays["y_pred_cycles"])
    np.save(os.path.join(REPORTS_DIR, "val_safe_rul_cycles.npy"), arrays["safe_rul"])
    np.save(os.path.join(REPORTS_DIR, "val_error_cycles.npy"), arrays["error"])
    np.save(os.path.join(REPORTS_DIR, "val_abs_error_cycles.npy"), arrays["abs_error"])
    np.save(os.path.join(REPORTS_DIR, "val_pred_fail_prob.npy"), arrays["y_pred_fail_prob"])
    np.save(os.path.join(REPORTS_DIR, "val_true_alarm.npy"), arrays["y_true_fail"])
    np.save(os.path.join(REPORTS_DIR, "val_neural_alarm.npy"), arrays["neural_alarm"])
    np.save(os.path.join(REPORTS_DIR, "val_safety_alarm.npy"), arrays["safety_alarm"])
    np.save(os.path.join(REPORTS_DIR, "val_hybrid_alarm.npy"), arrays["hybrid_alarm"])

    log_header("Saving model artifacts")

    final_model_path = os.path.join(MODELS_DIR, "fd004_specialist_final.keras")
    weights_path = os.path.join(MODELS_DIR, "fd004_specialist_weights.weights.h5")

    model.save(final_model_path)
    model.save_weights(weights_path)

    history_path = os.path.join(REPORTS_DIR, "final_history.json")
    save_json(history_path, history.history)

    save_training_report(
        val_metrics=val_metrics,
        diagnostics=diagnostics,
        training_config=training_config,
        threshold_selection={key: value for key, value in threshold_selection.items() if key != "rows"},
        metadata=metadata,
    )

    log_header("Saved artifacts")
    logger.info(f"Best checkpoint model: {os.path.join(MODELS_DIR, 'fd004_specialist.keras')}")
    logger.info(f"Final model: {final_model_path}")
    logger.info(f"Weights: {weights_path}")
    logger.info(f"Training history: {history_path}")
    logger.info(f"Validation diagnostics: {os.path.join(REPORTS_DIR, 'validation_results.json')}")
    logger.info(f"Selected threshold: {os.path.join(REPORTS_DIR, 'selected_failure_threshold.json')}")

    elapsed = time.time() - start_time

    log_header("PHASE 2 COMPLETED")
    logger.info(f"Elapsed time: {elapsed:.2f} seconds")
    logger.info(f"Reports directory: {REPORTS_DIR}")
    logger.info(f"Models directory: {MODELS_DIR}")


if __name__ == "__main__":
    main()