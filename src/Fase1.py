import os
import json
import time
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
import joblib

warnings.filterwarnings("ignore")

# =========================================================
# 1. CONFIGURATION
# =========================================================
RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"
REPORTS_DIR = os.path.join("reports", "phase_1_eda")

for directory in [PROCESSED_DIR, MODELS_DIR, REPORTS_DIR]:
    os.makedirs(directory, exist_ok=True)

COLUMNS = ["Engine_ID", "Cycle", "Set1", "Set2", "Set3"] + [f"S{i}" for i in range(1, 22)]
SETTINGS = ["Set1", "Set2", "Set3"]
SETTING_FEATURES = [f"{col}_std" for col in SETTINGS]

SENSORS_TO_USE = [
    "S2", "S3", "S4", "S7", "S8", "S9", "S11",
    "S12", "S13", "S14", "S15", "S17", "S20", "S21"
]

WINDOW_SIZE = 30
RUL_CAP = 125
FAIL_THRESHOLD = 30
N_CLUSTERS = 6
ROLLING_WINDOW = 5
TREND_LAG = 5
RANDOM_STATE = 42

CLIP_SCALED_FEATURES = True

LOG_FILE = os.path.join(REPORTS_DIR, "phase_1.log")


# =========================================================
# 2. LOGGER
# =========================================================
logger = logging.getLogger("phase1")
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


# =========================================================
# 3. UTILS
# =========================================================
def log_header(title: str):
    logger.info("")
    logger.info("=" * 100)
    logger.info(title)
    logger.info("=" * 100)


def ensure_sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["Engine_ID", "Cycle"]).reset_index(drop=True)


def validate_raw_frame(df: pd.DataFrame):
    required = set(COLUMNS)
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df.isna().any().any():
        na_counts = df.isna().sum()
        raise ValueError(f"Input data contains missing values. Counts:\n{na_counts[na_counts > 0]}")

    if (df["Cycle"] <= 0).any():
        raise ValueError("Cycle values must be strictly positive.")

    duplicated = df.duplicated(subset=["Engine_ID", "Cycle"]).sum()
    if duplicated > 0:
        raise ValueError(f"Duplicated Engine_ID/Cycle rows detected: {duplicated}")


def load_raw_dataset(path: str) -> pd.DataFrame:
    logger.info(f"Loading raw dataset from: {path}")
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    validate_raw_frame(df)
    return ensure_sorted(df)


def add_rul_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    max_cycles = df.groupby("Engine_ID", as_index=False)["Cycle"].max()
    max_cycles.rename(columns={"Cycle": "Max_Cycle"}, inplace=True)

    df = df.merge(max_cycles, on="Engine_ID", how="left")

    df["RUL"] = df["Max_Cycle"] - df["Cycle"]
    df["RUL_Piecewise"] = np.clip(df["RUL"], 0, RUL_CAP)
    df["Fail_Alarm"] = (df["RUL_Piecewise"] <= FAIL_THRESHOLD).astype(int)

    return df


def split_by_engine(df: pd.DataFrame, test_size: float = 0.2, random_state: int = RANDOM_STATE):
    engine_summary = (
        df.groupby("Engine_ID")
        .agg(Max_Cycle=("Cycle", "max"))
        .reset_index()
    )

    stratify_labels = None
    n_unique_bins = min(5, engine_summary["Max_Cycle"].nunique())

    if n_unique_bins >= 2:
        try:
            stratify_labels = pd.qcut(
                engine_summary["Max_Cycle"],
                q=n_unique_bins,
                duplicates="drop"
            ).astype(str)

            if stratify_labels.value_counts().min() < 2:
                logger.warning("Stratified split disabled because at least one bin has fewer than 2 engines.")
                stratify_labels = None

        except Exception as exc:
            logger.warning(f"Stratified split fallback activated due to qcut issue: {exc}")
            stratify_labels = None

    engine_ids = engine_summary["Engine_ID"].values

    train_ids, val_ids = train_test_split(
        engine_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels
    )

    train_df = df[df["Engine_ID"].isin(train_ids)].copy()
    val_df = df[df["Engine_ID"].isin(val_ids)].copy()

    train_df = ensure_sorted(train_df)
    val_df = ensure_sorted(val_df)

    overlap = set(train_df["Engine_ID"].unique()).intersection(set(val_df["Engine_ID"].unique()))
    if overlap:
        raise RuntimeError(f"Engine leakage detected between train and validation splits: {sorted(overlap)}")

    return train_df, val_df, engine_summary


def fit_operational_scaler(train_df: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[SETTINGS].astype(np.float64))
    return scaler


def add_standardized_settings(df: pd.DataFrame, settings_scaler: StandardScaler) -> pd.DataFrame:
    df = df.copy()
    scaled_settings = settings_scaler.transform(df[SETTINGS].astype(np.float64))

    for i, col in enumerate(SETTING_FEATURES):
        df[col] = scaled_settings[:, i].astype(np.float64)

    return df


def assign_clusters(df: pd.DataFrame, settings_scaler: StandardScaler, kmeans: KMeans) -> pd.DataFrame:
    df = df.copy()
    scaled_settings = settings_scaler.transform(df[SETTINGS].astype(np.float64))
    df["Cluster"] = kmeans.predict(scaled_settings)
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = ensure_sorted(df)

    for col in SENSORS_TO_USE:
        var_col = f"{col}_var"
        trend_col = f"{col}_trend"

        df[var_col] = (
            df.groupby("Engine_ID")[col]
            .transform(lambda s: s.rolling(window=ROLLING_WINDOW, min_periods=1).var())
            .fillna(0.0)
            .astype(np.float64)
        )

        df[trend_col] = (
            df.groupby("Engine_ID")[col]
            .transform(lambda s: s.diff(periods=TREND_LAG))
            .fillna(0.0)
            .astype(np.float64)
        )

    return df


def build_sensor_feature_list():
    features = list(SENSORS_TO_USE)

    for col in SENSORS_TO_USE:
        features.extend([f"{col}_var", f"{col}_trend"])

    return features


def build_feature_list():
    return build_sensor_feature_list() + SETTING_FEATURES


def fit_cluster_feature_scalers(train_df: pd.DataFrame, sensor_feature_cols: list):
    cluster_scalers = {}

    global_scaler = MinMaxScaler(feature_range=(0, 1))
    global_scaler.fit(train_df[sensor_feature_cols].astype(np.float64))

    for cluster_id in range(N_CLUSTERS):
        cluster_mask = train_df["Cluster"] == cluster_id
        n_samples = int(cluster_mask.sum())

        if n_samples >= 2:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(train_df.loc[cluster_mask, sensor_feature_cols].astype(np.float64))
            logger.info(f"[Cluster {cluster_id}] Feature scaler fitted on {n_samples} training samples.")
        else:
            scaler = global_scaler
            logger.warning(
                f"[Cluster {cluster_id}] Insufficient training samples. "
                f"Global fallback scaler will be used."
            )

        cluster_scalers[cluster_id] = scaler

    return cluster_scalers


def apply_cluster_feature_scaling(
    df: pd.DataFrame,
    sensor_feature_cols: list,
    cluster_scalers: dict,
    clip_scaled_features: bool = CLIP_SCALED_FEATURES
) -> pd.DataFrame:
    df = df.copy()
    df[sensor_feature_cols] = df[sensor_feature_cols].astype(np.float64)

    for cluster_id in range(N_CLUSTERS):
        mask = df["Cluster"] == cluster_id
        if mask.sum() == 0:
            continue

        scaler = cluster_scalers[cluster_id]
        scaled_values = scaler.transform(df.loc[mask, sensor_feature_cols])

        if clip_scaled_features:
            scaled_values = np.clip(scaled_values, 0.0, 1.0)

        df.loc[mask, sensor_feature_cols] = scaled_values

    return df


def summarize_split(df: pd.DataFrame, name: str) -> dict:
    return {
        "split": name,
        "rows": int(len(df)),
        "engines": int(df["Engine_ID"].nunique()),
        "cycles_min": int(df["Cycle"].min()),
        "cycles_max": int(df["Cycle"].max()),
        "rul_min": float(df["RUL_Piecewise"].min()) if "RUL_Piecewise" in df.columns else None,
        "rul_max": float(df["RUL_Piecewise"].max()) if "RUL_Piecewise" in df.columns else None,
        "failure_rate_rows": float(df["Fail_Alarm"].mean()) if "Fail_Alarm" in df.columns else None,
    }


def summarize_clusters(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Cluster")
        .agg(
            samples=("Cluster", "size"),
            engines=("Engine_ID", "nunique"),
            mean_cycle=("Cycle", "mean"),
            mean_rul=("RUL_Piecewise", "mean"),
            failure_rate=("Fail_Alarm", "mean"),
        )
        .reset_index()
        .sort_values("Cluster")
    )


def create_sequences(df: pd.DataFrame, feature_cols: list, window_size: int = WINDOW_SIZE):
    X, y_rul, y_fail = [], [], []
    engine_ids, cycle_ids = [], []

    for eng_id in df["Engine_ID"].unique():
        eng_data = df[df["Engine_ID"] == eng_id].copy()
        eng_data = ensure_sorted(eng_data)

        if len(eng_data) < window_size:
            continue

        feat_vals = eng_data[feature_cols].values.astype(np.float32)
        rul_vals = eng_data["RUL_Piecewise"].values.astype(np.float32)
        fail_vals = eng_data["Fail_Alarm"].values.astype(np.float32)
        cycles = eng_data["Cycle"].values.astype(np.int32)

        for end_idx in range(window_size, len(eng_data) + 1):
            start_idx = end_idx - window_size

            X.append(feat_vals[start_idx:end_idx, :])
            y_rul.append(rul_vals[end_idx - 1] / RUL_CAP)
            y_fail.append(fail_vals[end_idx - 1])
            engine_ids.append(eng_id)
            cycle_ids.append(cycles[end_idx - 1])

    return (
        np.array(X, dtype=np.float32),
        np.array(y_rul, dtype=np.float32),
        np.array(y_fail, dtype=np.float32),
        np.array(engine_ids, dtype=np.int32),
        np.array(cycle_ids, dtype=np.int32),
    )


def create_official_test_tensor(df: pd.DataFrame, feature_cols: list, window_size: int = WINDOW_SIZE):
    X_test = []
    engine_ids = []

    for eng_id in df["Engine_ID"].unique():
        eng_data = df[df["Engine_ID"] == eng_id].copy()
        eng_data = ensure_sorted(eng_data)

        feat_vals = eng_data[feature_cols].values.astype(np.float32)

        if len(feat_vals) >= window_size:
            seq = feat_vals[-window_size:]
        else:
            seq = np.pad(
                feat_vals,
                ((window_size - len(feat_vals), 0), (0, 0)),
                mode="edge"
            )

        X_test.append(seq)
        engine_ids.append(eng_id)

    return np.array(X_test, dtype=np.float32), np.array(engine_ids, dtype=np.int32)


def save_official_test_targets():
    rul_test_path = os.path.join(RAW_DIR, "RUL_FD004.txt")

    if not os.path.exists(rul_test_path):
        logger.warning("Official RUL_FD004.txt not found. Test targets will not be saved.")
        return None

    rul_test = pd.read_csv(rul_test_path, sep=r"\s+", header=None, names=["RUL"])
    rul_test["RUL_Piecewise"] = np.clip(rul_test["RUL"], 0, RUL_CAP)
    rul_test["RUL_Normalized"] = rul_test["RUL_Piecewise"] / RUL_CAP
    rul_test["Fail_Alarm"] = (rul_test["RUL_Piecewise"] <= FAIL_THRESHOLD).astype(int)

    csv_path = os.path.join(PROCESSED_DIR, "official_test_targets_fd004.csv")
    npy_rul_path = os.path.join(PROCESSED_DIR, "y_test_official_fd004_rul.npy")
    npy_alarm_path = os.path.join(PROCESSED_DIR, "y_test_official_fd004_fail.npy")

    rul_test.to_csv(csv_path, index=False)
    np.save(npy_rul_path, rul_test["RUL_Normalized"].values.astype(np.float32))
    np.save(npy_alarm_path, rul_test["Fail_Alarm"].values.astype(np.float32))

    logger.info(f"Official test targets saved to: {csv_path}")
    logger.info(f"Official normalized RUL target saved to: {npy_rul_path}")
    logger.info(f"Official failure target saved to: {npy_alarm_path}")

    return {
        "csv": csv_path,
        "rul_npy": npy_rul_path,
        "alarm_npy": npy_alarm_path,
        "test_target_rows": int(len(rul_test)),
        "test_target_fail_rate": float(rul_test["Fail_Alarm"].mean()),
    }


def validate_final_features(df: pd.DataFrame, feature_cols: list, name: str):
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing final feature columns: {missing}")

    values = df[feature_cols].values

    if not np.isfinite(values).all():
        raise ValueError(f"[{name}] Final features contain NaN or Inf.")

    logger.info(
        f"[{name}] Final feature range: min={float(np.min(values)):.6f}, "
        f"max={float(np.max(values)):.6f}, mean={float(np.mean(values)):.6f}, "
        f"std={float(np.std(values)):.6f}"
    )


# =========================================================
# 4. PLOTS
# =========================================================
def plot_engine_length_distribution(engine_summary: pd.DataFrame, train_ids, val_ids):
    log_header("Plot 1 - Engine Length Distribution")

    plt.figure(figsize=(11, 6), dpi=300)

    train_lengths = engine_summary[engine_summary["Engine_ID"].isin(train_ids)]["Max_Cycle"]
    val_lengths = engine_summary[engine_summary["Engine_ID"].isin(val_ids)]["Max_Cycle"]

    sns.histplot(train_lengths, bins=20, kde=True, stat="density", label="Training engines", alpha=0.55)
    sns.histplot(val_lengths, bins=20, kde=True, stat="density", label="Validation engines", alpha=0.45)

    plt.title("Distribution of Engine Life Lengths by Split", fontweight="bold")
    plt.xlabel("Maximum Cycle Count per Engine", fontweight="bold")
    plt.ylabel("Density", fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "01_Engine_Length_Distribution.png"))
    plt.close()


def plot_operational_settings(train_df: pd.DataFrame):
    log_header("Plot 2 - Operational Regime Scatter Plot")

    plt.figure(figsize=(10, 8), dpi=300)

    scatter = plt.scatter(
        train_df["Set1_std"],
        train_df["Set2_std"],
        c=train_df["Cluster"],
        cmap="tab10",
        s=18,
        alpha=0.75
    )

    plt.title("Operational Regime Distribution in Standardized Setting Space", fontweight="bold")
    plt.xlabel("Set1 standardized", fontweight="bold")
    plt.ylabel("Set2 standardized", fontweight="bold")

    cbar = plt.colorbar(scatter)
    cbar.set_label("Operational Cluster", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "02_Operational_Regime_Scatter.png"))
    plt.close()


def plot_cluster_distribution(train_df: pd.DataFrame, val_df: pd.DataFrame):
    log_header("Plot 3 - Cluster Allocation Across Splits")

    train_counts = train_df["Cluster"].value_counts().sort_index()
    val_counts = val_df["Cluster"].value_counts().sort_index()

    cluster_index = list(range(N_CLUSTERS))
    train_vals = [int(train_counts.get(i, 0)) for i in cluster_index]
    val_vals = [int(val_counts.get(i, 0)) for i in cluster_index]

    x = np.arange(N_CLUSTERS)
    width = 0.38

    plt.figure(figsize=(10, 6), dpi=300)

    plt.bar(x - width / 2, train_vals, width, label="Training split")
    plt.bar(x + width / 2, val_vals, width, label="Validation split")

    plt.title("Operational Cluster Allocation Across Train and Validation Sets", fontweight="bold")
    plt.xlabel("Cluster Identifier", fontweight="bold")
    plt.ylabel("Number of Samples", fontweight="bold")
    plt.xticks(x, [f"C{i}" for i in cluster_index])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "03_Cluster_Allocation.png"))
    plt.close()


def plot_rul_distribution(train_df: pd.DataFrame, val_df: pd.DataFrame):
    log_header("Plot 4 - Piecewise RUL Distribution")

    plt.figure(figsize=(11, 6), dpi=300)

    sns.kdeplot(train_df["RUL_Piecewise"], fill=True, label="Training RUL", alpha=0.4)
    sns.kdeplot(val_df["RUL_Piecewise"], fill=True, label="Validation RUL", alpha=0.4)

    plt.axvline(FAIL_THRESHOLD, linestyle="--", linewidth=2, label=f"Failure threshold = {FAIL_THRESHOLD} cycles")
    plt.axvline(RUL_CAP, linestyle=":", linewidth=2, label=f"RUL cap = {RUL_CAP} cycles")

    plt.title("Piecewise RUL Distribution Across Dataset Splits", fontweight="bold")
    plt.xlabel("Piecewise RUL cycles", fontweight="bold")
    plt.ylabel("Density", fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "04_RUL_Distribution.png"))
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, feature_cols: list):
    log_header("Plot 5 - Correlation Matrix")

    corr_cols = feature_cols + ["RUL_Piecewise"]
    corr_matrix = df[corr_cols].corr()

    plt.figure(figsize=(18, 15), dpi=300)
    sns.heatmap(
        corr_matrix,
        annot=False,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5
    )

    plt.title("Pearson Correlation Matrix: Engineered Features, Settings and Piecewise RUL", fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "05_Correlation_Matrix.png"))
    plt.close()


def plot_representative_degradation(df: pd.DataFrame):
    log_header("Plot 6 - Representative Engine Degradation Profile")

    engine_lengths = df.groupby("Engine_ID")["Cycle"].max().sort_values()
    sample_engine_id = int(engine_lengths.index[len(engine_lengths) // 2])
    eng = df[df["Engine_ID"] == sample_engine_id].copy()

    corr_cols = SENSORS_TO_USE + ["RUL_Piecewise"]
    top_sensors = (
        df[corr_cols].corr()["RUL_Piecewise"]
        .drop("RUL_Piecewise")
        .abs()
        .sort_values(ascending=False)
        .head(3)
        .index.tolist()
    )

    fig, ax1 = plt.subplots(figsize=(14, 7), dpi=300)
    ax2 = ax1.twinx()

    ax1.plot(
        eng["Cycle"],
        eng["RUL_Piecewise"],
        "k--",
        linewidth=2.5,
        label="Piecewise RUL target"
    )

    ax1.set_xlabel("Flight cycles", fontweight="bold")
    ax1.set_ylabel("Remaining useful life cycles", fontweight="bold", color="k")

    for sensor in top_sensors:
        smoothed = eng[sensor].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
        ax2.plot(
            eng["Cycle"],
            smoothed,
            linewidth=2.2,
            alpha=0.9,
            label=f"{sensor} rolling mean"
        )

    ax2.set_ylabel("Scaled sensor value", fontweight="bold")

    plt.title(
        f"Representative Degradation Profile and RUL Trajectory Engine {sample_engine_id}",
        fontweight="bold",
        pad=15
    )

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "06_Degradation_Profile.png"))
    plt.close()


def plot_feature_variability(train_df: pd.DataFrame, feature_cols: list):
    log_header("Plot 7 - Feature Variability Snapshot")

    top_var_features = (
        train_df[feature_cols]
        .var()
        .sort_values(ascending=False)
        .head(10)
        .index.tolist()
    )

    plt.figure(figsize=(13, 7), dpi=300)
    sns.boxplot(data=train_df[top_var_features], orient="h")
    plt.title("Distribution of the Highest-Variance Final Features", fontweight="bold")
    plt.xlabel("Feature value", fontweight="bold")
    plt.ylabel("Feature name", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "07_High_Variance_Features.png"))
    plt.close()


def plot_missing_values_overview(df: pd.DataFrame):
    log_header("Plot 8 - Missing Values Overview")

    plt.figure(figsize=(14, 4), dpi=300)

    missing_pct = df.isna().mean().sort_values(ascending=False) * 100
    missing_pct = missing_pct[missing_pct > 0]

    if len(missing_pct) == 0:
        plt.text(0.5, 0.5, "No missing values detected.", ha="center", va="center", fontsize=14)
        plt.axis("off")
    else:
        sns.barplot(x=missing_pct.index, y=missing_pct.values)
        plt.xticks(rotation=90)
        plt.ylabel("Missing values percentage")
        plt.title("Missing Values Overview")

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "08_Missing_Values_Overview.png"))
    plt.close()


# =========================================================
# 5. DOCUMENTATION / REPORT
# =========================================================
def dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available._"

    headers = list(df.columns)
    lines = []

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for _, row in df.iterrows():
        formatted = []

        for value in row.tolist():
            if isinstance(value, (float, np.floating)):
                formatted.append(f"{value:.6f}")
            else:
                formatted.append(str(value))

        lines.append("| " + " | ".join(formatted) + " |")

    return "\n".join(lines)


def save_phase_report(
    train_summary,
    val_summary,
    feature_cols,
    sensor_feature_cols,
    sequence_shapes,
    model_paths,
    cluster_summary: pd.DataFrame,
    test_target_info
):
    report_path = os.path.join(REPORTS_DIR, "phase_1_report.md")
    cluster_md = dataframe_to_markdown_table(cluster_summary)

    test_target_text = "_Official test targets were not found._"
    if test_target_info is not None:
        test_target_text = f"""
- Official test target rows: {test_target_info["test_target_rows"]}
- Official test failure rate: {test_target_info["test_target_fail_rate"]:.6f}
- Official normalized RUL target: `{test_target_info["rul_npy"]}`
- Official failure target: `{test_target_info["alarm_npy"]}`
"""

    content = f"""# Phase 1 — FD004 Preprocessing and Exploratory Data Analysis Report

## Objective
Prepare a reproducible and leakage-safe preprocessing pipeline for Remaining Useful Life prediction on the CMAPSS FD004 dataset.

## Dataset Overview
- Training engines: {train_summary["engines"]}
- Validation engines: {val_summary["engines"]}
- Training rows: {train_summary["rows"]}
- Validation rows: {val_summary["rows"]}
- Window size: {WINDOW_SIZE}
- Piecewise RUL cap: {RUL_CAP}
- Failure threshold: {FAIL_THRESHOLD}
- Operational clusters: {N_CLUSTERS}

## Feature Engineering
- Raw sensor channels used: {len(SENSORS_TO_USE)}
- Engineered features per sensor: rolling variance and lagged trend
- Sensor and temporal feature dimension: {len(sensor_feature_cols)}
- Standardized operational setting features: {len(SETTING_FEATURES)}
- Final feature dimension: {len(feature_cols)}
- Temporal descriptors are computed before final cluster-specific feature scaling
- Operational settings are preserved as standardized model inputs
- Cluster-specific MinMax scaling is fitted only on the training split

## Sequence Shapes
- X_train: {sequence_shapes["X_train"]}
- X_val: {sequence_shapes["X_val"]}
- X_test_official: {sequence_shapes["X_test"]}

## Cluster Summary

{cluster_md}

## Official Test Targets

{test_target_text}

## Saved Artifacts
- Settings scaler: `{model_paths["settings_scaler"]}`
- KMeans model: `{model_paths["kmeans"]}`
- Cluster feature scalers: `{model_paths["cluster_scalers"]}`
- Training tensor: `{model_paths["X_train"]}`
- Validation tensor: `{model_paths["X_val"]}`
- Official test tensor: `{model_paths["X_test"]}`

## Notes
The train-validation split is performed by engine identifier to avoid temporal and engine-level leakage.
Operational clusters represent operating regimes, not engine groups: each engine can pass through multiple operating conditions.
This phase prepares the data for CNN-LSTM multi-task RUL regression and failure-state classification.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"Markdown report saved to: {report_path}")


# =========================================================
# 6. MAIN PIPELINE
# =========================================================
def main():
    start_time = time.time()

    log_header("PHASE 1 - FD004 ADVANCED PREPROCESSING PIPELINE")
    logger.info("Starting data preparation workflow.")

    train_path = os.path.join(RAW_DIR, "train_FD004.txt")
    test_path = os.path.join(RAW_DIR, "test_FD004.txt")

    df = load_raw_dataset(train_path)

    logger.info(f"Training dataset loaded successfully: {len(df)} rows, {df['Engine_ID'].nunique()} engines.")
    logger.info(f"Operational settings columns: {SETTINGS}")
    logger.info(f"Selected sensor channels: {SENSORS_TO_USE}")

    df = add_rul_targets(df)
    logger.info("RUL, piecewise RUL, and failure alarm labels were created successfully.")

    train_df, val_df, engine_summary = split_by_engine(df, test_size=0.20, random_state=RANDOM_STATE)

    logger.info(f"Training split: {train_df['Engine_ID'].nunique()} engines, {len(train_df)} rows.")
    logger.info(f"Validation split: {val_df['Engine_ID'].nunique()} engines, {len(val_df)} rows.")

    train_summary = summarize_split(train_df, "train")
    val_summary = summarize_split(val_df, "validation")

    pd.DataFrame([train_summary, val_summary]).to_csv(
        os.path.join(REPORTS_DIR, "split_summary.csv"),
        index=False
    )

    logger.info("Split summary exported to CSV.")
    logger.info(f"Training row-level failure rate: {train_summary['failure_rate_rows']:.6f}")
    logger.info(f"Validation row-level failure rate: {val_summary['failure_rate_rows']:.6f}")

    log_header("Operational Regime Modeling")

    settings_scaler = fit_operational_scaler(train_df)

    train_settings_scaled = settings_scaler.transform(train_df[SETTINGS].astype(np.float64))

    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=20
    )

    train_clusters = kmeans.fit_predict(train_settings_scaled)

    train_df = train_df.copy()
    train_df["Cluster"] = train_clusters

    val_df = assign_clusters(val_df, settings_scaler, kmeans)

    train_df = add_standardized_settings(train_df, settings_scaler)
    val_df = add_standardized_settings(val_df, settings_scaler)

    logger.info("Cluster assignment completed for training and validation splits.")
    logger.info("Standardized operational settings added as model features.")
    logger.info("Cluster counts on training split:\n" + str(train_df["Cluster"].value_counts().sort_index().to_dict()))
    logger.info("Cluster counts on validation split:\n" + str(val_df["Cluster"].value_counts().sort_index().to_dict()))

    log_header("Feature Engineering")

    train_df = add_engineered_features(train_df)
    val_df = add_engineered_features(val_df)

    sensor_feature_cols = build_sensor_feature_list()
    feature_cols = build_feature_list()

    logger.info(f"Sensor and temporal feature columns: {len(sensor_feature_cols)}")
    logger.info(f"Final feature columns including operational settings: {len(feature_cols)}")
    logger.info(f"Feature columns: {feature_cols}")

    cluster_scalers = fit_cluster_feature_scalers(train_df, sensor_feature_cols)

    logger.info("Applying cluster-specific feature scaling to training and validation sets.")

    train_df = apply_cluster_feature_scaling(train_df, sensor_feature_cols, cluster_scalers)
    val_df = apply_cluster_feature_scaling(val_df, sensor_feature_cols, cluster_scalers)

    validate_final_features(train_df, feature_cols, "train")
    validate_final_features(val_df, feature_cols, "validation")

    cluster_summary = summarize_clusters(train_df)
    logger.info("Cluster summary on training data:\n" + cluster_summary.to_string(index=False))

    log_header("Diagnostic Visualization")

    plot_engine_length_distribution(engine_summary, train_df["Engine_ID"].unique(), val_df["Engine_ID"].unique())
    plot_operational_settings(train_df)
    plot_cluster_distribution(train_df, val_df)
    plot_rul_distribution(train_df, val_df)
    plot_correlation_heatmap(train_df, feature_cols)
    plot_representative_degradation(train_df)
    plot_feature_variability(train_df, feature_cols)
    plot_missing_values_overview(df)

    log_header("Sequence Construction")

    X_train, y_train_rul, y_train_fail, train_engine_ids, train_cycle_ids = create_sequences(
        train_df, feature_cols, window_size=WINDOW_SIZE
    )

    X_val, y_val_rul, y_val_fail, val_engine_ids, val_cycle_ids = create_sequences(
        val_df, feature_cols, window_size=WINDOW_SIZE
    )

    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_val shape: {X_val.shape}")
    logger.info(f"y_train_rul shape: {y_train_rul.shape}")
    logger.info(f"y_val_rul shape: {y_val_rul.shape}")
    logger.info(f"y_train_fail positive rate: {float(y_train_fail.mean()):.6f}")
    logger.info(f"y_val_fail positive rate: {float(y_val_fail.mean()):.6f}")

    np.save(os.path.join(PROCESSED_DIR, "X_train_fd004.npy"), X_train)
    np.save(os.path.join(PROCESSED_DIR, "y_train_fd004_rul.npy"), y_train_rul)
    np.save(os.path.join(PROCESSED_DIR, "y_train_fd004_fail.npy"), y_train_fail)

    np.save(os.path.join(PROCESSED_DIR, "X_val_fd004.npy"), X_val)
    np.save(os.path.join(PROCESSED_DIR, "y_val_fd004_rul.npy"), y_val_rul)
    np.save(os.path.join(PROCESSED_DIR, "y_val_fd004_fail.npy"), y_val_fail)

    np.save(os.path.join(PROCESSED_DIR, "train_engine_ids.npy"), train_engine_ids)
    np.save(os.path.join(PROCESSED_DIR, "train_cycle_ids.npy"), train_cycle_ids)
    np.save(os.path.join(PROCESSED_DIR, "val_engine_ids.npy"), val_engine_ids)
    np.save(os.path.join(PROCESSED_DIR, "val_cycle_ids.npy"), val_cycle_ids)

    logger.info("Training and validation tensors saved successfully.")

    log_header("Official Test Set Processing")

    test_df = load_raw_dataset(test_path)

    logger.info(f"Official test dataset loaded successfully: {len(test_df)} rows, {test_df['Engine_ID'].nunique()} engines.")

    test_df = assign_clusters(test_df, settings_scaler, kmeans)
    test_df = add_standardized_settings(test_df, settings_scaler)
    test_df = add_engineered_features(test_df)
    test_df = apply_cluster_feature_scaling(test_df, sensor_feature_cols, cluster_scalers)

    validate_final_features(test_df, feature_cols, "official_test")

    X_test, test_engine_ids = create_official_test_tensor(test_df, feature_cols, window_size=WINDOW_SIZE)

    np.save(os.path.join(PROCESSED_DIR, "X_test_official_fd004.npy"), X_test)
    np.save(os.path.join(PROCESSED_DIR, "test_engine_ids.npy"), test_engine_ids)

    logger.info(f"Official test tensor shape: {X_test.shape}")
    logger.info("Official test tensor saved successfully.")

    test_target_info = save_official_test_targets()

    settings_scaler_path = os.path.join(MODELS_DIR, "fd004_settings_scaler.joblib")
    kmeans_path = os.path.join(MODELS_DIR, "fd004_kmeans.joblib")
    cluster_scalers_path = os.path.join(MODELS_DIR, "fd004_cluster_scalers.joblib")

    joblib.dump(settings_scaler, settings_scaler_path)
    joblib.dump(kmeans, kmeans_path)
    joblib.dump(cluster_scalers, cluster_scalers_path)

    logger.info("Models and scalers saved successfully.")

    metadata = {
        "dataset": "CMAPSS FD004",
        "window_size": WINDOW_SIZE,
        "rul_cap": RUL_CAP,
        "failure_threshold": FAIL_THRESHOLD,
        "n_clusters": N_CLUSTERS,
        "rolling_window": ROLLING_WINDOW,
        "trend_lag": TREND_LAG,
        "clip_scaled_features": CLIP_SCALED_FEATURES,
        "feature_columns": feature_cols,
        "sensor_feature_columns": sensor_feature_cols,
        "setting_feature_columns": SETTING_FEATURES,
        "sensor_columns": SENSORS_TO_USE,
        "settings_columns": SETTINGS,
        "final_feature_count": len(feature_cols),
        "train_sequences": int(X_train.shape[0]),
        "val_sequences": int(X_val.shape[0]),
        "test_sequences": int(X_test.shape[0]),
        "train_fail_rate": float(y_train_fail.mean()),
        "val_fail_rate": float(y_val_fail.mean()),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_engines": int(train_df["Engine_ID"].nunique()),
        "val_engines": int(val_df["Engine_ID"].nunique()),
        "test_engines": int(test_df["Engine_ID"].nunique()),
        "settings_scaler_path": settings_scaler_path,
        "kmeans_path": kmeans_path,
        "cluster_scalers_path": cluster_scalers_path,
    }

    if test_target_info is not None:
        metadata.update(test_target_info)

    metadata_path = os.path.join(PROCESSED_DIR, "phase_1_metadata.json")

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"Metadata saved to: {metadata_path}")

    save_phase_report(
        train_summary=train_summary,
        val_summary=val_summary,
        feature_cols=feature_cols,
        sensor_feature_cols=sensor_feature_cols,
        sequence_shapes={
            "X_train": list(X_train.shape),
            "X_val": list(X_val.shape),
            "X_test": list(X_test.shape),
        },
        model_paths={
            "settings_scaler": settings_scaler_path,
            "kmeans": kmeans_path,
            "cluster_scalers": cluster_scalers_path,
            "X_train": os.path.join(PROCESSED_DIR, "X_train_fd004.npy"),
            "X_val": os.path.join(PROCESSED_DIR, "X_val_fd004.npy"),
            "X_test": os.path.join(PROCESSED_DIR, "X_test_official_fd004.npy"),
        },
        cluster_summary=cluster_summary,
        test_target_info=test_target_info,
    )

    elapsed = time.time() - start_time

    log_header("PHASE 1 COMPLETED")
    logger.info(f"Elapsed time: {elapsed:.2f} seconds")
    logger.info(f"Reports directory: {REPORTS_DIR}")
    logger.info(f"Processed directory: {PROCESSED_DIR}")
    logger.info(f"Models directory: {MODELS_DIR}")


if __name__ == "__main__":
    main()