# Phase 1 — FD004 Preprocessing and Exploratory Data Analysis Report

## Objective
Prepare a reproducible and leakage-safe preprocessing pipeline for Remaining Useful Life prediction on the CMAPSS FD004 dataset.

## Dataset Overview
- Training engines: 199
- Validation engines: 50
- Training rows: 48692
- Validation rows: 12557
- Window size: 30
- Piecewise RUL cap: 125
- Failure threshold: 30
- Operational clusters: 6

## Feature Engineering
- Raw sensor channels used: 14
- Engineered features per sensor: rolling variance and lagged trend
- Sensor and temporal feature dimension: 42
- Standardized operational setting features: 3
- Final feature dimension: 45
- Temporal descriptors are computed before final cluster-specific feature scaling
- Operational settings are preserved as standardized model inputs
- Cluster-specific MinMax scaling is fitted only on the training split

## Sequence Shapes
- X_train: [42921, 30, 45]
- X_val: [11107, 30, 45]
- X_test_official: [248, 30, 45]

## Cluster Summary

| Cluster | samples | engines | mean_cycle | mean_rul | failure_rate |
| --- | --- | --- | --- | --- | --- |
| 0.000000 | 7352.000000 | 199.000000 | 132.944097 | 93.156420 | 0.126632 |
| 1.000000 | 7299.000000 | 199.000000 | 134.281271 | 92.883272 | 0.125634 |
| 2.000000 | 7256.000000 | 199.000000 | 133.018330 | 93.481946 | 0.122244 |
| 3.000000 | 7208.000000 | 199.000000 | 134.079079 | 92.662597 | 0.127913 |
| 4.000000 | 7332.000000 | 199.000000 | 133.147845 | 92.282324 | 0.126568 |
| 5.000000 | 12245.000000 | 199.000000 | 133.377297 | 92.584973 | 0.129359 |

## Official Test Targets


- Official test target rows: 248
- Official test failure rate: 0.213710
- Official normalized RUL target: `data/processed/y_test_official_fd004_rul.npy`
- Official failure target: `data/processed/y_test_official_fd004_fail.npy`


## Saved Artifacts
- Settings scaler: `models/fd004_settings_scaler.joblib`
- KMeans model: `models/fd004_kmeans.joblib`
- Cluster feature scalers: `models/fd004_cluster_scalers.joblib`
- Training tensor: `data/processed/X_train_fd004.npy`
- Validation tensor: `data/processed/X_val_fd004.npy`
- Official test tensor: `data/processed/X_test_official_fd004.npy`

## Notes
The train-validation split is performed by engine identifier to avoid temporal and engine-level leakage.
Operational clusters represent operating regimes, not engine groups: each engine can pass through multiple operating conditions.
This phase prepares the data for CNN-LSTM multi-task RUL regression and failure-state classification.
