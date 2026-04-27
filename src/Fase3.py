import os
import time
import json
import logging
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import load_model

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
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


# ---------------------------------------------------------
# SYSTEM CONFIGURATION
# ---------------------------------------------------------
# IMPORTANT: keep GPU disabled on Mac.
tf.config.set_visible_devices([], "GPU")

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"
PHASE2_REPORTS_DIR = os.path.join("reports", "phase_2_training")
REPORTS_DIR = os.path.join("reports", "phase_3_evaluation")

os.makedirs(REPORTS_DIR, exist_ok=True)

LOG_FILE = os.path.join(REPORTS_DIR, "phase_3_diagnostics.log")

RUL_CAP = 125.0
FAIL_THRESHOLD = 30.0
SAFETY_MARGIN = 15.0
DEFAULT_FAIL_PROB_THRESHOLD = 0.50
RANDOM_STATE = 42


# ---------------------------------------------------------
# LOGGER
# ---------------------------------------------------------
logger = logging.getLogger("phase3")
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


def log_header(title: str):
    logger.info("")
    logger.info("=" * 100)
    logger.info(title)
    logger.info("=" * 100)


def log_dict(title: str, values: dict):
    logger.info(title)
    for key, value in values.items():
        logger.info(f"  - {key}: {value}")


# ---------------------------------------------------------
# LOADING FUNCTIONS
# ---------------------------------------------------------
def load_metadata():
    metadata_path = os.path.join(PROCESSED_DIR, "phase_1_metadata.json")

    if not os.path.exists(metadata_path):
        logger.warning("Phase 1 metadata file was not found.")
        return {}

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    logger.info(f"Metadata loaded from: {metadata_path}")
    return metadata


def load_validation_selected_threshold():
    threshold_path = os.path.join(PHASE2_REPORTS_DIR, "selected_failure_threshold.json")

    if not os.path.exists(threshold_path):
        logger.warning("Validation-selected threshold not found. Falling back to default threshold 0.50.")
        return DEFAULT_FAIL_PROB_THRESHOLD, {
            "selected_threshold": DEFAULT_FAIL_PROB_THRESHOLD,
            "selected_strategy": "default_fallback",
            "selection_source": "fallback",
        }

    with open(threshold_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    selected_threshold = float(payload.get("selected_threshold", DEFAULT_FAIL_PROB_THRESHOLD))

    logger.info(f"Validation-selected threshold loaded from: {threshold_path}")
    logger.info(f"Selected validation threshold: {selected_threshold}")

    return selected_threshold, payload


def resolve_model_path():
    candidates = [
        os.path.join(MODELS_DIR, "fd004_specialist.keras"),
        os.path.join(MODELS_DIR, "fd004_specialist_final.keras"),
    ]

    for path in candidates:
        if os.path.exists(path):
            logger.info(f"Selected model artifact: {path}")
            return path

    raise FileNotFoundError(
        "No trained model was found. Expected fd004_specialist.keras or fd004_specialist_final.keras."
    )


def load_official_targets():
    processed_target_path = os.path.join(PROCESSED_DIR, "official_test_targets_fd004.csv")
    raw_target_path = os.path.join(RAW_DIR, "RUL_FD004.txt")

    if os.path.exists(processed_target_path):
        target_df = pd.read_csv(processed_target_path)

        if "RUL" in target_df.columns:
            y_true_raw = target_df["RUL"].values.astype(np.float32)
        elif "True_RUL_Raw" in target_df.columns:
            y_true_raw = target_df["True_RUL_Raw"].values.astype(np.float32)
        else:
            raise ValueError(f"Processed target file exists but does not contain a valid RUL column: {processed_target_path}")

        logger.info(f"Official test targets loaded from processed file: {processed_target_path}")
        return y_true_raw

    if os.path.exists(raw_target_path):
        y_true_raw = pd.read_csv(raw_target_path, sep=r"\s+", header=None).values.flatten().astype(np.float32)
        logger.info(f"Official test targets loaded from raw file: {raw_target_path}")
        return y_true_raw

    raise FileNotFoundError("Missing official test targets. Expected official_test_targets_fd004.csv or RUL_FD004.txt.")


def load_evaluation_inputs():
    model_path = resolve_model_path()

    tensor_path = os.path.join(PROCESSED_DIR, "X_test_official_fd004.npy")
    engine_ids_path = os.path.join(PROCESSED_DIR, "test_engine_ids.npy")

    missing = [path for path in [tensor_path, engine_ids_path] if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing evaluation artifacts: {missing}")

    model = load_model(model_path, compile=False)
    X_test = np.load(tensor_path).astype(np.float32)
    engine_ids = np.load(engine_ids_path).astype(np.int32)
    y_true_raw = load_official_targets()

    if len(y_true_raw) != X_test.shape[0]:
        raise ValueError(
            f"Ground-truth length mismatch. RUL has {len(y_true_raw)} rows, "
            f"but X_test has {X_test.shape[0]} sequences."
        )

    if len(engine_ids) != X_test.shape[0]:
        raise ValueError(
            f"Engine ID length mismatch. Engine ID array has {len(engine_ids)} rows, "
            f"but X_test has {X_test.shape[0]} sequences."
        )

    return model, X_test, engine_ids, y_true_raw, model_path


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------
def compute_regression_metrics(y_true, y_pred, prefix=""):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    error = y_pred - y_true
    abs_error = np.abs(error)

    return {
        f"{prefix}MAE": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        f"{prefix}R2": float(r2_score(y_true, y_pred)),
        f"{prefix}Bias": float(np.mean(error)),
        f"{prefix}Median_Absolute_Error": float(np.median(abs_error)),
        f"{prefix}Dangerous_Overestimation_Pct": float(np.mean(error > 0.0) * 100.0),
        f"{prefix}Underestimation_Pct": float(np.mean(error < 0.0) * 100.0),
        f"{prefix}Within_5_Cycles_Pct": float(np.mean(abs_error <= 5.0) * 100.0),
        f"{prefix}Within_10_Cycles_Pct": float(np.mean(abs_error <= 10.0) * 100.0),
        f"{prefix}Within_15_Cycles_Pct": float(np.mean(abs_error <= 15.0) * 100.0),
    }


def compute_alarm_metrics(y_true_alarm, y_pred_alarm, score=None, prefix=""):
    cm = confusion_matrix(y_true_alarm, y_pred_alarm, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        f"{prefix}Accuracy": float(accuracy_score(y_true_alarm, y_pred_alarm)),
        f"{prefix}Precision": float(precision_score(y_true_alarm, y_pred_alarm, zero_division=0)),
        f"{prefix}Recall": float(recall_score(y_true_alarm, y_pred_alarm, zero_division=0)),
        f"{prefix}F1": float(f1_score(y_true_alarm, y_pred_alarm, zero_division=0)),
        f"{prefix}TN": int(tn),
        f"{prefix}FP": int(fp),
        f"{prefix}FN": int(fn),
        f"{prefix}TP": int(tp),
    }

    if score is not None and len(np.unique(y_true_alarm)) == 2:
        metrics[f"{prefix}ROC_AUC"] = float(roc_auc_score(y_true_alarm, score))
        metrics[f"{prefix}Average_Precision"] = float(average_precision_score(y_true_alarm, score))

    return metrics, cm


def find_best_threshold_by_f1(y_true_alarm, scores):
    thresholds = np.linspace(0.01, 0.99, 99)

    best = {
        "threshold": 0.5,
        "f1": -1.0,
        "precision": 0.0,
        "recall": 0.0,
        "accuracy": 0.0,
    }

    rows = []

    for threshold in thresholds:
        pred = (scores >= threshold).astype(int)

        row = {
            "threshold": float(threshold),
            "precision": float(precision_score(y_true_alarm, pred, zero_division=0)),
            "recall": float(recall_score(y_true_alarm, pred, zero_division=0)),
            "f1": float(f1_score(y_true_alarm, pred, zero_division=0)),
            "accuracy": float(accuracy_score(y_true_alarm, pred)),
        }

        rows.append(row)

        if row["f1"] > best["f1"]:
            best = row.copy()

    return best, rows


def nasa_phm_score(y_true, y_pred):
    """
    NASA PHM-style asymmetric scoring function.
    Positive error means late prediction / dangerous overestimation.
    Negative error means early prediction / conservative underestimation.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    errors = y_pred - y_true
    score = 0.0

    for error in errors:
        if error < 0:
            score += np.exp(-error / 13.0) - 1.0
        else:
            score += np.exp(error / 10.0) - 1.0

    return float(score)


# ---------------------------------------------------------
# PLOTS
# ---------------------------------------------------------
def save_plot(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, filename))
    plt.close()
    logger.info(f"Plot saved: {os.path.join(REPORTS_DIR, filename)}")


def plot_prediction_scatter(eval_df, metrics):
    log_header("Plot 1 - RUL prediction scatter")

    plt.figure(figsize=(10, 8), dpi=300)

    scatter = plt.scatter(
        eval_df["True_RUL"],
        eval_df["Pred_RUL"],
        c=eval_df["NN_Fail_Prob"],
        cmap="coolwarm",
        alpha=0.82,
        edgecolor="k",
        s=50,
    )

    plt.plot([0, RUL_CAP], [0, RUL_CAP], "k--", linewidth=2, label="Ideal prediction")
    plt.fill_between(
        [0, RUL_CAP],
        [-15, RUL_CAP - 15],
        [15, RUL_CAP + 15],
        alpha=0.12,
        label="±15 cycle band"
    )

    plt.axvline(FAIL_THRESHOLD, linestyle=":", linewidth=2, label=f"Failure threshold = {FAIL_THRESHOLD:.0f}")
    plt.colorbar(scatter, label="Neural failure probability")

    plt.title(
        f"Official Test Set: True RUL vs Predicted RUL\n"
        f"MAE={metrics['Global_MAE']:.2f}, RMSE={metrics['Global_RMSE']:.2f}, R²={metrics['Global_R2']:.3f}",
        fontweight="bold"
    )
    plt.xlabel("True RUL cycles", fontweight="bold")
    plt.ylabel("Predicted RUL cycles", fontweight="bold")
    plt.xlim(0, RUL_CAP)
    plt.ylim(0, RUL_CAP)
    plt.legend(loc="upper left")

    save_plot("01_RUL_Prediction_Scatter.png")


def plot_error_distribution(eval_df):
    log_header("Plot 2 - Error distribution")

    plt.figure(figsize=(10, 6), dpi=300)

    sns.kdeplot(eval_df["Error"], fill=True, alpha=0.45, label="Raw model error")
    sns.kdeplot(eval_df["Safe_Error"], fill=True, alpha=0.45, label=f"Safety-adjusted error (-{SAFETY_MARGIN:.0f} cycles)")

    plt.axvline(0, linestyle="--", linewidth=2, label="Zero error")
    plt.title("Prediction Error Distribution", fontweight="bold")
    plt.xlabel("Error = Predicted RUL - True RUL cycles", fontweight="bold")
    plt.ylabel("Density", fontweight="bold")
    plt.legend()

    save_plot("02_Error_Distribution.png")


def plot_abs_error_by_rul(eval_df):
    log_header("Plot 3 - Absolute error by true RUL")

    plt.figure(figsize=(10, 6), dpi=300)

    plt.scatter(eval_df["True_RUL"], eval_df["Abs_Error"], alpha=0.7)
    sns.regplot(
        x=eval_df["True_RUL"],
        y=eval_df["Abs_Error"],
        scatter=False,
        line_kws={"linestyle": "--", "linewidth": 2}
    )

    plt.axvline(FAIL_THRESHOLD, linestyle=":", linewidth=2, label=f"Failure threshold = {FAIL_THRESHOLD:.0f}")
    plt.gca().invert_xaxis()

    plt.title("Absolute RUL Error as Failure Approaches", fontweight="bold")
    plt.xlabel("True RUL cycles; lower means closer to failure", fontweight="bold")
    plt.ylabel("Absolute error cycles", fontweight="bold")
    plt.legend()

    save_plot("03_Absolute_Error_vs_RUL.png")


def plot_confusion_matrix(cm, title, filename):
    log_header(f"Plot - {title}")

    plt.figure(figsize=(8, 6), dpi=300)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted healthy", "Predicted critical"],
        yticklabels=["True healthy", "True critical"],
        annot_kws={"size": 15, "weight": "bold"},
    )

    plt.title(title, fontweight="bold")
    plt.xlabel("Predicted state", fontweight="bold")
    plt.ylabel("True state", fontweight="bold")

    save_plot(filename)


def plot_roc_curve(y_true_alarm, scores, filename, title):
    log_header(f"Plot - {title}")

    if len(np.unique(y_true_alarm)) < 2:
        logger.warning("ROC curve skipped because only one class is present.")
        return

    fpr, tpr, _ = roc_curve(y_true_alarm, scores)
    auc_value = roc_auc_score(y_true_alarm, scores)

    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC AUC = {auc_value:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Random baseline")

    plt.title(title, fontweight="bold")
    plt.xlabel("False positive rate", fontweight="bold")
    plt.ylabel("True positive rate", fontweight="bold")
    plt.legend()

    save_plot(filename)


def plot_precision_recall_curve(y_true_alarm, scores, filename, title):
    log_header(f"Plot - {title}")

    if len(np.unique(y_true_alarm)) < 2:
        logger.warning("Precision-recall curve skipped because only one class is present.")
        return

    precision, recall, _ = precision_recall_curve(y_true_alarm, scores)
    ap_value = average_precision_score(y_true_alarm, scores)

    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(recall, precision, linewidth=2, label=f"Average precision = {ap_value:.3f}")

    plt.title(title, fontweight="bold")
    plt.xlabel("Recall", fontweight="bold")
    plt.ylabel("Precision", fontweight="bold")
    plt.legend()

    save_plot(filename)


def plot_risk_distribution(eval_df):
    log_header("Plot 8 - Operational risk distribution")

    plt.figure(figsize=(10, 6), dpi=300)

    bins = np.histogram_bin_edges(eval_df["Safe_Error"], bins=30)

    plt.hist(
        eval_df.loc[eval_df["Safe_Error"] < 0, "Safe_Error"],
        bins=bins,
        alpha=0.75,
        label="Conservative underestimation"
    )

    plt.hist(
        eval_df.loc[eval_df["Safe_Error"] >= 0, "Safe_Error"],
        bins=bins,
        alpha=0.75,
        label="Dangerous overestimation"
    )

    plt.axvline(0, linewidth=2)
    plt.title("Operational Risk Distribution After Safety Margin", fontweight="bold")
    plt.xlabel("Safety-adjusted error cycles", fontweight="bold")
    plt.ylabel("Number of engines", fontweight="bold")
    plt.legend()

    save_plot("08_Operational_Risk_Distribution.png")


def plot_top_error_engines(eval_df):
    log_header("Plot 9 - Highest-error engines")

    top_df = eval_df.sort_values("Abs_Error", ascending=False).head(15).copy()
    top_df["Engine_Label"] = top_df["Engine_ID"].astype(str)

    plt.figure(figsize=(12, 7), dpi=300)
    sns.barplot(data=top_df, x="Abs_Error", y="Engine_Label", orient="h")

    plt.title("Top 15 Engines by Absolute RUL Prediction Error", fontweight="bold")
    plt.xlabel("Absolute error cycles", fontweight="bold")
    plt.ylabel("Engine ID", fontweight="bold")

    save_plot("09_Top_Error_Engines.png")


def plot_probability_distribution(eval_df, validation_threshold):
    log_header("Plot 10 - Failure probability distribution")

    plt.figure(figsize=(10, 6), dpi=300)

    sns.histplot(
        data=eval_df,
        x="NN_Fail_Prob",
        hue="True_Alarm",
        bins=30,
        stat="density",
        common_norm=False,
        alpha=0.55
    )

    plt.axvline(
        validation_threshold,
        linestyle="--",
        linewidth=2,
        label=f"Validation threshold = {validation_threshold:.2f}"
    )

    plt.title("Neural Failure Probability Distribution by True Alarm State", fontweight="bold")
    plt.xlabel("Neural failure probability", fontweight="bold")
    plt.ylabel("Density", fontweight="bold")
    plt.legend()

    save_plot("10_Failure_Probability_Distribution.png")


def plot_threshold_sweep(rows, validation_threshold, posthoc_best_threshold):
    log_header("Plot 11 - Threshold sweep")

    sweep_df = pd.DataFrame(rows)
    sweep_df.to_csv(os.path.join(REPORTS_DIR, "threshold_sweep_neural_failure_posthoc.csv"), index=False)

    plt.figure(figsize=(10, 6), dpi=300)

    plt.plot(sweep_df["threshold"], sweep_df["precision"], linewidth=2, label="Precision")
    plt.plot(sweep_df["threshold"], sweep_df["recall"], linewidth=2, label="Recall")
    plt.plot(sweep_df["threshold"], sweep_df["f1"], linewidth=2, label="F1-score")

    plt.axvline(validation_threshold, linestyle="--", linewidth=1.5, label="Validation-selected threshold")
    plt.axvline(posthoc_best_threshold, linestyle=":", linewidth=1.5, label="Post-hoc best F1 threshold")

    plt.title("Neural Failure Classifier Threshold Sweep on Official Test", fontweight="bold")
    plt.xlabel("Decision threshold", fontweight="bold")
    plt.ylabel("Metric value", fontweight="bold")
    plt.legend()

    save_plot("11_Neural_Threshold_Sweep_Posthoc.png")


def plot_metric_comparison(summary_metrics):
    log_header("Plot 12 - Alarm strategy comparison")

    strategies = [
        {
            "strategy": "Validation threshold neural",
            "accuracy": summary_metrics["Validation_Threshold_Neural_Accuracy"],
            "precision": summary_metrics["Validation_Threshold_Neural_Precision"],
            "recall": summary_metrics["Validation_Threshold_Neural_Recall"],
            "f1": summary_metrics["Validation_Threshold_Neural_F1"],
        },
        {
            "strategy": "Safety RUL rule",
            "accuracy": summary_metrics["Safety_Accuracy"],
            "precision": summary_metrics["Safety_Precision"],
            "recall": summary_metrics["Safety_Recall"],
            "f1": summary_metrics["Safety_F1"],
        },
        {
            "strategy": "Hybrid OR rule",
            "accuracy": summary_metrics["Hybrid_Accuracy"],
            "precision": summary_metrics["Hybrid_Precision"],
            "recall": summary_metrics["Hybrid_Recall"],
            "f1": summary_metrics["Hybrid_F1"],
        },
        {
            "strategy": "Post-hoc best F1 neural",
            "accuracy": summary_metrics["Posthoc_Neural_Accuracy"],
            "precision": summary_metrics["Posthoc_Neural_Precision"],
            "recall": summary_metrics["Posthoc_Neural_Recall"],
            "f1": summary_metrics["Posthoc_Neural_F1"],
        },
    ]

    metric_df = pd.DataFrame(strategies)
    metric_df.to_csv(os.path.join(REPORTS_DIR, "alarm_strategy_comparison.csv"), index=False)

    long_df = metric_df.melt(id_vars="strategy", var_name="metric", value_name="value")

    plt.figure(figsize=(12, 7), dpi=300)
    sns.barplot(data=long_df, x="metric", y="value", hue="strategy")

    plt.title("Alarm Strategy Comparison on Official Test Set", fontweight="bold")
    plt.xlabel("Metric", fontweight="bold")
    plt.ylabel("Score", fontweight="bold")
    plt.ylim(0, 1.05)
    plt.legend(title="Strategy", fontsize=8)

    save_plot("12_Alarm_Strategy_Comparison.png")


# ---------------------------------------------------------
# REPORTING
# ---------------------------------------------------------
def dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available._"

    headers = list(df.columns)
    lines = []

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for _, row in df.iterrows():
        cells = []

        for value in row.tolist():
            if isinstance(value, (float, np.floating)):
                cells.append(f"{float(value):.6f}")
            else:
                cells.append(str(value))

        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def save_json(path: str, payload: dict):
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


def save_markdown_report(summary_metrics, model_path, validation_threshold_payload, posthoc_best_threshold, metadata):
    report_path = os.path.join(REPORTS_DIR, "phase_3_report.md")

    metric_rows = []
    for key, value in summary_metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            metric_rows.append({"Metric": key, "Value": float(value)})

    metric_df = pd.DataFrame(metric_rows)

    validation_threshold = float(validation_threshold_payload.get("selected_threshold", DEFAULT_FAIL_PROB_THRESHOLD))
    validation_strategy = validation_threshold_payload.get("selected_strategy", "unknown")

    content = f"""# Phase 3 — FD004 Official Evaluation and Prognostic Diagnostics Report

## Objective
Evaluate the trained multi-task FD004 prognostic model on the official CMAPSS test set.

This phase uses the failure threshold selected during validation, avoiding test-set threshold tuning.

## Model Artifact
- Evaluated model: `{model_path}`

## Phase 1 Input
- Window size: {metadata.get("window_size", "unknown")}
- Feature count: {metadata.get("final_feature_count", "unknown")}

## Evaluation Configuration
- RUL cap: {RUL_CAP:.0f} cycles
- Failure threshold: {FAIL_THRESHOLD:.0f} cycles
- Safety margin: {SAFETY_MARGIN:.0f} cycles
- Validation-selected neural threshold: {validation_threshold:.4f}
- Validation threshold strategy: `{validation_strategy}`
- Post-hoc test best-F1 threshold: {posthoc_best_threshold["threshold"]:.4f}

## Main Metrics

{dataframe_to_markdown_table(metric_df)}

## Interpretation Notes
- `Pred_RUL` is the raw model RUL prediction clipped to the valid range.
- `Safe_RUL` is an operational safety rule obtained by subtracting the safety margin.
- Main regression accuracy must be interpreted using `Pred_RUL`, not `Safe_RUL`.
- The post-hoc threshold is reported only for diagnostic comparison and is not used as the official operating threshold.

## Generated Diagnostic Files
- `Full_Engine_Predictions.csv`
- `Final_Metrics.json`
- `alarm_strategy_comparison.csv`
- `threshold_sweep_neural_failure_posthoc.csv`
- `01_RUL_Prediction_Scatter.png`
- `02_Error_Distribution.png`
- `03_Absolute_Error_vs_RUL.png`
- `04_Validation_Threshold_Neural_Confusion_Matrix.png`
- `05_Safety_RUL_Confusion_Matrix.png`
- `06_Hybrid_Confusion_Matrix.png`
- `07_Neural_ROC_Curve.png`
- `07b_Neural_Precision_Recall_Curve.png`
- `08_Operational_Risk_Distribution.png`
- `09_Top_Error_Engines.png`
- `10_Failure_Probability_Distribution.png`
- `11_Neural_Threshold_Sweep_Posthoc.png`
- `12_Alarm_Strategy_Comparison.png`
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"Markdown report saved to: {report_path}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    start_time = time.time()

    log_header("PHASE 3 - OFFICIAL FD004 EVALUATION AND DIAGNOSTICS")

    metadata = load_metadata()
    validation_threshold, validation_threshold_payload = load_validation_selected_threshold()

    if metadata:
        log_dict("Loaded Phase 1 metadata", {
            "window_size": metadata.get("window_size"),
            "feature_count": metadata.get("final_feature_count"),
            "test_sequences": metadata.get("test_sequences"),
            "train_fail_rate": metadata.get("train_fail_rate"),
            "val_fail_rate": metadata.get("val_fail_rate"),
            "test_target_fail_rate": metadata.get("test_target_fail_rate"),
        })

    model, X_test, engine_ids, y_true_raw, model_path = load_evaluation_inputs()

    log_header("Input diagnostics")
    log_dict("Official test tensor summary", {
        "X_test shape": X_test.shape,
        "number of engines": len(engine_ids),
        "RUL ground-truth rows": len(y_true_raw),
        "raw true RUL min": float(np.min(y_true_raw)),
        "raw true RUL max": float(np.max(y_true_raw)),
        "validation_selected_threshold": validation_threshold,
    })

    y_true_piecewise = np.clip(y_true_raw, 0.0, RUL_CAP)
    y_true_alarm = (y_true_piecewise <= FAIL_THRESHOLD).astype(int)

    logger.info("Running deterministic inference on official test tensor.")

    X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
    predictions = model(X_test_tensor, training=False)

    if not isinstance(predictions, (list, tuple)) or len(predictions) != 2:
        raise RuntimeError("The loaded model must produce exactly two outputs: RUL and failure probability.")

    y_pred_rul_raw = predictions[0].numpy().reshape(-1) * RUL_CAP
    y_pred_fail_prob = predictions[1].numpy().reshape(-1)

    y_pred_rul = np.clip(y_pred_rul_raw, 0.0, RUL_CAP)
    y_safe_rul = np.clip(y_pred_rul - SAFETY_MARGIN, 0.0, RUL_CAP)

    validation_threshold_neural_alarm = (y_pred_fail_prob >= validation_threshold).astype(int)
    safety_alarm = (y_safe_rul <= FAIL_THRESHOLD).astype(int)
    hybrid_alarm = np.logical_or(validation_threshold_neural_alarm == 1, safety_alarm == 1).astype(int)

    posthoc_best_threshold, threshold_rows = find_best_threshold_by_f1(y_true_alarm, y_pred_fail_prob)
    posthoc_neural_alarm = (y_pred_fail_prob >= posthoc_best_threshold["threshold"]).astype(int)

    eval_df = pd.DataFrame({
        "Engine_ID": engine_ids,
        "True_RUL_Raw": y_true_raw,
        "True_RUL": y_true_piecewise,
        "Pred_RUL_Raw": y_pred_rul_raw,
        "Pred_RUL": y_pred_rul,
        "Safe_RUL": y_safe_rul,
        "NN_Fail_Prob": y_pred_fail_prob,
        "True_Alarm": y_true_alarm,
        "Validation_Threshold_Neural_Alarm": validation_threshold_neural_alarm,
        "Safety_Alarm": safety_alarm,
        "Hybrid_Alarm": hybrid_alarm,
        "Posthoc_Neural_Alarm": posthoc_neural_alarm,
    })

    eval_df["Error"] = eval_df["Pred_RUL"] - eval_df["True_RUL"]
    eval_df["Abs_Error"] = eval_df["Error"].abs()
    eval_df["Safe_Error"] = eval_df["Safe_RUL"] - eval_df["True_RUL"]
    eval_df["Dangerous_Overestimation"] = (eval_df["Error"] > 0.0).astype(int)
    eval_df["Safe_Dangerous_Overestimation"] = (eval_df["Safe_Error"] > 0.0).astype(int)

    eval_csv_path = os.path.join(REPORTS_DIR, "Full_Engine_Predictions.csv")
    eval_df.to_csv(eval_csv_path, index=False)
    logger.info(f"Full engine-level prediction table saved to: {eval_csv_path}")

    log_header("Regression metrics")

    regression_metrics = compute_regression_metrics(
        eval_df["True_RUL"],
        eval_df["Pred_RUL"],
        prefix="Global_"
    )

    safe_regression_metrics = compute_regression_metrics(
        eval_df["True_RUL"],
        eval_df["Safe_RUL"],
        prefix="Safety_Adjusted_"
    )

    range_metrics = {}

    ranges = {
        "High_RUL_": eval_df["True_RUL"] > 80.0,
        "Mid_RUL_": (eval_df["True_RUL"] > FAIL_THRESHOLD) & (eval_df["True_RUL"] <= 80.0),
        "End_of_Life_": eval_df["True_RUL"] <= FAIL_THRESHOLD,
    }

    for prefix, mask in ranges.items():
        if int(mask.sum()) > 0:
            range_metrics.update(
                compute_regression_metrics(
                    eval_df.loc[mask, "True_RUL"],
                    eval_df.loc[mask, "Pred_RUL"],
                    prefix=prefix
                )
            )
            range_metrics[f"{prefix}Samples"] = int(mask.sum())
        else:
            range_metrics[f"{prefix}Samples"] = 0

    phm_score_raw = nasa_phm_score(eval_df["True_RUL"].values, eval_df["Pred_RUL"].values)
    phm_score_safe = nasa_phm_score(eval_df["True_RUL"].values, eval_df["Safe_RUL"].values)

    log_dict("Raw RUL regression metrics", regression_metrics)
    log_dict("Safety-adjusted RUL regression metrics", safe_regression_metrics)
    log_dict("Range-specific RUL metrics", range_metrics)
    logger.info(f"NASA PHM asymmetric score - raw prediction: {phm_score_raw:.6f}")
    logger.info(f"NASA PHM asymmetric score - safety-adjusted prediction: {phm_score_safe:.6f}")

    log_header("Alarm metrics")

    validation_neural_metrics, validation_neural_cm = compute_alarm_metrics(
        y_true_alarm,
        validation_threshold_neural_alarm,
        score=y_pred_fail_prob,
        prefix="Validation_Threshold_Neural_"
    )

    safety_metrics, safety_cm = compute_alarm_metrics(
        y_true_alarm,
        safety_alarm,
        score=-y_safe_rul,
        prefix="Safety_"
    )

    hybrid_metrics, hybrid_cm = compute_alarm_metrics(
        y_true_alarm,
        hybrid_alarm,
        score=np.maximum(y_pred_fail_prob, (RUL_CAP - y_safe_rul) / RUL_CAP),
        prefix="Hybrid_"
    )

    posthoc_neural_metrics, posthoc_neural_cm = compute_alarm_metrics(
        y_true_alarm,
        posthoc_neural_alarm,
        score=y_pred_fail_prob,
        prefix="Posthoc_Neural_"
    )

    log_dict("Validation-threshold neural alarm metrics", validation_neural_metrics)
    log_dict("Safety-adjusted RUL alarm metrics", safety_metrics)
    log_dict("Hybrid OR alarm metrics", hybrid_metrics)
    log_dict("Post-hoc neural alarm metrics", posthoc_neural_metrics)
    log_dict("Post-hoc best-F1 neural threshold", posthoc_best_threshold)

    summary_metrics = {}

    summary_metrics.update(regression_metrics)
    summary_metrics.update(safe_regression_metrics)
    summary_metrics.update(range_metrics)
    summary_metrics.update(validation_neural_metrics)
    summary_metrics.update(safety_metrics)
    summary_metrics.update(hybrid_metrics)
    summary_metrics.update(posthoc_neural_metrics)

    summary_metrics["NASA_PHM_Score_Raw"] = phm_score_raw
    summary_metrics["NASA_PHM_Score_Safety_Adjusted"] = phm_score_safe
    summary_metrics["Safety_Margin"] = SAFETY_MARGIN
    summary_metrics["Validation_Selected_Fail_Probability_Threshold"] = validation_threshold
    summary_metrics["Posthoc_Test_Optimized_Fail_Probability_Threshold"] = posthoc_best_threshold["threshold"]

    metrics_path = os.path.join(REPORTS_DIR, "Final_Metrics.json")
    save_json(metrics_path, summary_metrics)

    np.savetxt(
        os.path.join(REPORTS_DIR, "validation_threshold_neural_confusion_matrix.csv"),
        validation_neural_cm,
        delimiter=",",
        fmt="%d"
    )

    np.savetxt(
        os.path.join(REPORTS_DIR, "safety_confusion_matrix.csv"),
        safety_cm,
        delimiter=",",
        fmt="%d"
    )

    np.savetxt(
        os.path.join(REPORTS_DIR, "hybrid_confusion_matrix.csv"),
        hybrid_cm,
        delimiter=",",
        fmt="%d"
    )

    np.savetxt(
        os.path.join(REPORTS_DIR, "posthoc_neural_confusion_matrix.csv"),
        posthoc_neural_cm,
        delimiter=",",
        fmt="%d"
    )

    log_header("Generating diagnostic plots")

    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("paper", font_scale=1.1)

    plot_prediction_scatter(eval_df, regression_metrics)
    plot_error_distribution(eval_df)
    plot_abs_error_by_rul(eval_df)

    plot_confusion_matrix(
        validation_neural_cm,
        "Validation-Threshold Neural Classifier Confusion Matrix",
        "04_Validation_Threshold_Neural_Confusion_Matrix.png"
    )

    plot_confusion_matrix(
        safety_cm,
        "Safety-Adjusted RUL Rule Confusion Matrix",
        "05_Safety_RUL_Confusion_Matrix.png"
    )

    plot_confusion_matrix(
        hybrid_cm,
        "Hybrid OR Alarm Strategy Confusion Matrix",
        "06_Hybrid_Confusion_Matrix.png"
    )

    plot_roc_curve(
        y_true_alarm,
        y_pred_fail_prob,
        "07_Neural_ROC_Curve.png",
        "Neural Failure Classifier ROC Curve"
    )

    plot_precision_recall_curve(
        y_true_alarm,
        y_pred_fail_prob,
        "07b_Neural_Precision_Recall_Curve.png",
        "Neural Failure Classifier Precision-Recall Curve"
    )

    plot_risk_distribution(eval_df)
    plot_top_error_engines(eval_df)
    plot_probability_distribution(eval_df, validation_threshold)
    plot_threshold_sweep(threshold_rows, validation_threshold, posthoc_best_threshold["threshold"])
    plot_metric_comparison(summary_metrics)

    save_markdown_report(
        summary_metrics=summary_metrics,
        model_path=model_path,
        validation_threshold_payload=validation_threshold_payload,
        posthoc_best_threshold=posthoc_best_threshold,
        metadata=metadata
    )

    elapsed = time.time() - start_time

    log_header("PHASE 3 COMPLETED")
    logger.info(f"Elapsed time: {elapsed:.2f} seconds")
    logger.info(f"Reports directory: {REPORTS_DIR}")
    logger.info(f"Log file: {LOG_FILE}")


if __name__ == "__main__":
    main()