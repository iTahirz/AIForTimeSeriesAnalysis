# Risk-Aware Predictive Maintenance for Turbofan Engines ✈️🔧

This repository contains the full pipeline for predicting the **Remaining Useful Life (RUL)** and the probability of critical failure of aircraft engines, using the highly complex NASA CMAPSS FD004 dataset.

## 🎯 Project Objective
The goal is to develop a predictive model that inherently prioritizes flight safety. Overestimating the RUL of an engine can lead to catastrophic failures, while underestimating it only results in early (safe) maintenance. This project implements a **Conservative Asymmetric Loss** and a **Hybrid Decision Logic** to aggressively minimize dangerous overestimations.

## ⚙️ Technical Architecture
1. **Physics-Informed Preprocessing (Phase 1):** - **K-Means Clustering (k=6)** to decode the flight envelope (Altitude, Mach Number) dynamically.
   - Cluster-specific MinMax scaling to eliminate environmental noise from sensors.
   - Extraction of 5-cycle rolling variance and lagged trends.
2. **Multi-Task CNN-LSTM (Phase 2):** - A hybrid architecture combining 1D-CNNs for spatial feature extraction and LSTMs for temporal degradation tracking.
   - The network branches into two heads: a regression head (predicting exact RUL cycles) and a classification head (predicting imminent failure probability).
3. **Hybrid Alarm Strategy (Phase 3):** - An OR-Logic combining the neural classifier (threshold optimized for >95% Recall) and a Safety-Adjusted RUL prediction (-15 cycles margin).

## 🚀 Key Results
Evaluated on the official test set (248 unseen engines), the Hybrid Strategy achieved:
- **Recall: 98.1%** (Successfully identified 52 out of 53 critical failures).
- **False Negatives:** Only 1.
- **R² Score:** 0.775 (Global tracking).
- **End-of-Life MAE:** 6.4 cycles (High precision exactly when failure is imminent).

## 📁 Repository Structure
- `src/`: Python scripts for data preparation, training, and evaluation.
- `reports/`: Automatically generated Markdown reports, confusion matrices, and diagnostic plots (`.png`).
- *(Note: Raw data, trained models, and massive 3D tensors are excluded via `.gitignore` to maintain repository speed and clarity).*
