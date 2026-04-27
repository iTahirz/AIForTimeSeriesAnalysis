# Phase 2 — FD004 Conservative Balanced Multi-task Training Report

## Objective
Train a CNN-LSTM multi-task prognostic model for RUL regression and failure-state classification.

This version keeps the original stable architecture and only adjusts the training objective to reduce excessive conservative bias.

## Phase 1 Input
- Window size: 30
- Feature count: 45

## Conservative Loss
- EOL weight alpha: 3.2
- EOL weight beta: 3.5
- Overestimation alpha: 1.6
- Overestimation beta: 2.5
- Underestimation weight: 1.0

## Training Configuration
- Epochs: 150
- Batch size: 128
- Learning rate: 0.0005
- Gradient clipnorm: 1.0
- Dropout rate: 0.2
- RUL output loss weight: 4.0
- Failure output loss weight: 0.75
- Scheduler monitor: val_rul_output_mae
- Scheduler factor: 0.85
- Scheduler patience: 8
- Early stopping monitor: val_rul_output_mae
- GPU disabled: True

## Final Validation Metrics

| Metric | Value |
| --- | --- |
| fail_output_auc | 0.939471 |
| fail_output_binary_accuracy | 0.933375 |
| fail_output_loss | 0.981254 |
| fail_output_precision | 0.733834 |
| fail_output_recall | 0.820000 |
| loss | 0.976168 |
| rul_output_loss | 0.059030 |
| rul_output_mae | 0.137511 |
| rul_output_rmse | 0.194090 |

## Validation Diagnostics

| Metric | Value |
| --- | --- |
| validation_global_mae_cycles | 17.175621 |
| validation_global_rmse_cycles | 24.258436 |
| validation_global_bias_cycles | -6.279216 |
| validation_global_median_abs_error_cycles | 10.654167 |
| validation_global_overestimation_pct | 30.314216 |
| validation_global_underestimation_pct | 69.055551 |
| validation_global_within_5_cycles_pct | 19.069056 |
| validation_global_within_10_cycles_pct | 46.466192 |
| validation_global_within_15_cycles_pct | 62.258035 |
| validation_safety_adjusted_mae_cycles | 22.481220 |
| validation_safety_adjusted_rmse_cycles | 28.470209 |
| validation_safety_adjusted_bias_cycles | -16.165607 |
| validation_safety_adjusted_median_abs_error_cycles | 18.844612 |
| validation_safety_adjusted_overestimation_pct | 17.025299 |
| validation_safety_adjusted_underestimation_pct | 82.578554 |
| validation_safety_adjusted_within_5_cycles_pct | 10.452868 |
| validation_safety_adjusted_within_10_cycles_pct | 21.004772 |
| validation_safety_adjusted_within_15_cycles_pct | 33.276312 |
| validation_eol_mae_cycles | 8.162979 |
| validation_eol_rmse_cycles | 12.386714 |
| validation_eol_bias_cycles | 5.603951 |
| validation_eol_median_abs_error_cycles | 5.164966 |
| validation_eol_overestimation_pct | 71.161290 |
| validation_eol_underestimation_pct | 28.838710 |
| validation_eol_within_5_cycles_pct | 48.838710 |
| validation_eol_within_10_cycles_pct | 74.000000 |
| validation_eol_within_15_cycles_pct | 83.548387 |
| validation_eol_samples | 1550 |
| validation_mid_rul_mae_cycles | 22.410799 |
| validation_mid_rul_rmse_cycles | 28.208488 |
| validation_mid_rul_bias_cycles | 8.397931 |
| validation_mid_rul_median_abs_error_cycles | 18.212036 |
| validation_mid_rul_overestimation_pct | 56.480000 |
| validation_mid_rul_underestimation_pct | 43.520000 |
| validation_mid_rul_within_5_cycles_pct | 16.240000 |
| validation_mid_rul_within_10_cycles_pct | 29.520000 |
| validation_mid_rul_within_15_cycles_pct | 41.800000 |
| validation_mid_rul_samples | 2500 |
| validation_high_rul_mae_cycles | 17.300552 |
| validation_high_rul_rmse_cycles | 24.710426 |
| validation_high_rul_bias_cycles | -14.088735 |
| validation_high_rul_median_abs_error_cycles | 10.513069 |
| validation_high_rul_overestimation_pct | 12.073119 |
| validation_high_rul_underestimation_pct | 86.934958 |
| validation_high_rul_within_5_cycles_pct | 13.532663 |
| validation_high_rul_within_10_cycles_pct | 46.421992 |
| validation_high_rul_within_15_cycles_pct | 64.829248 |
| validation_high_rul_samples | 7057 |
| validation_neural_alarm_accuracy | 0.918430 |
| validation_neural_alarm_precision | 0.646898 |
| validation_neural_alarm_recall | 0.914839 |
| validation_neural_alarm_f1 | 0.757883 |
| validation_neural_alarm_tn | 8783 |
| validation_neural_alarm_fp | 774 |
| validation_neural_alarm_fn | 132 |
| validation_neural_alarm_tp | 1418 |
| validation_neural_alarm_roc_auc | 0.972211 |
| validation_neural_alarm_average_precision | 0.838818 |
| validation_safety_alarm_accuracy | 0.914198 |
| validation_safety_alarm_precision | 0.632844 |
| validation_safety_alarm_recall | 0.917419 |
| validation_safety_alarm_f1 | 0.749012 |
| validation_safety_alarm_tn | 8732 |
| validation_safety_alarm_fp | 825 |
| validation_safety_alarm_fn | 128 |
| validation_safety_alarm_tp | 1422 |
| validation_safety_alarm_roc_auc | 0.972242 |
| validation_safety_alarm_average_precision | 0.842488 |
| validation_hybrid_alarm_accuracy | 0.909607 |
| validation_hybrid_alarm_precision | 0.617571 |
| validation_hybrid_alarm_recall | 0.925161 |
| validation_hybrid_alarm_f1 | 0.740702 |
| validation_hybrid_alarm_tn | 8669 |
| validation_hybrid_alarm_fp | 888 |
| validation_hybrid_alarm_fn | 116 |
| validation_hybrid_alarm_tp | 1434 |
| validation_hybrid_alarm_roc_auc | 0.971879 |
| validation_hybrid_alarm_average_precision | 0.840864 |
| selected_failure_probability_threshold | 0.010000 |
| validation_safety_margin_cycles | 10.000000 |

## Validation Threshold Selection

| Strategy | Threshold | Precision | Recall | F1 | Accuracy |
| --- | --- | --- | --- | --- | --- |
| best_f1 | 0.440000 | 0.732685 | 0.825806 | 0.776463 | 0.933645 |
| best_precision_with_recall_at_least_90 | 0.010000 | 0.646898 | 0.914839 | 0.757883 | 0.918430 |

## Notes
Keras RUL metrics are normalized. Cycle-level diagnostics multiply predictions by 125.
The selected failure threshold is chosen on validation only and should be reused unchanged during official test evaluation.
