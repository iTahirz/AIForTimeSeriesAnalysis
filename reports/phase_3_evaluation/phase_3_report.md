# Phase 3 — FD004 Official Evaluation and Prognostic Diagnostics Report

## Objective
Evaluate the trained multi-task FD004 prognostic model on the official CMAPSS test set.

This phase uses the failure threshold selected during validation, avoiding test-set threshold tuning.

## Model Artifact
- Evaluated model: `models/fd004_specialist.keras`

## Phase 1 Input
- Window size: 30
- Feature count: 45

## Evaluation Configuration
- RUL cap: 125 cycles
- Failure threshold: 30 cycles
- Safety margin: 15 cycles
- Validation-selected neural threshold: 0.0100
- Validation threshold strategy: `best_precision_with_recall_at_least_90`
- Post-hoc test best-F1 threshold: 0.2400

## Main Metrics

| Metric | Value |
| --- | --- |
| Global_MAE | 16.959539 |
| Global_RMSE | 22.736928 |
| Global_R2 | 0.720161 |
| Global_Bias | -1.451428 |
| Global_Median_Absolute_Error | 11.364979 |
| Global_Dangerous_Overestimation_Pct | 45.161290 |
| Global_Underestimation_Pct | 54.435484 |
| Global_Within_5_Cycles_Pct | 19.354839 |
| Global_Within_10_Cycles_Pct | 39.112903 |
| Global_Within_15_Cycles_Pct | 60.080645 |
| Safety_Adjusted_MAE | 22.659258 |
| Safety_Adjusted_RMSE | 27.932574 |
| Safety_Adjusted_R2 | 0.577656 |
| Safety_Adjusted_Bias | -16.248653 |
| Safety_Adjusted_Median_Absolute_Error | 21.359657 |
| Safety_Adjusted_Dangerous_Overestimation_Pct | 19.758065 |
| Safety_Adjusted_Underestimation_Pct | 80.241935 |
| Safety_Adjusted_Within_5_Cycles_Pct | 9.677419 |
| Safety_Adjusted_Within_10_Cycles_Pct | 24.596774 |
| Safety_Adjusted_Within_15_Cycles_Pct | 38.306452 |
| High_RUL_MAE | 17.753725 |
| High_RUL_RMSE | 23.081987 |
| High_RUL_R2 | -1.438225 |
| High_RUL_Bias | -11.490809 |
| High_RUL_Median_Absolute_Error | 12.100456 |
| High_RUL_Dangerous_Overestimation_Pct | 22.222222 |
| High_RUL_Underestimation_Pct | 77.037037 |
| High_RUL_Within_5_Cycles_Pct | 9.629630 |
| High_RUL_Within_10_Cycles_Pct | 31.111111 |
| High_RUL_Within_15_Cycles_Pct | 59.259259 |
| High_RUL_Samples | 135.000000 |
| Mid_RUL_MAE | 23.121984 |
| Mid_RUL_RMSE | 28.954773 |
| Mid_RUL_R2 | -2.443934 |
| Mid_RUL_Bias | 14.119492 |
| Mid_RUL_Median_Absolute_Error | 19.902630 |
| Mid_RUL_Dangerous_Overestimation_Pct | 63.333333 |
| Mid_RUL_Underestimation_Pct | 36.666667 |
| Mid_RUL_Within_5_Cycles_Pct | 20.000000 |
| Mid_RUL_Within_10_Cycles_Pct | 31.666667 |
| Mid_RUL_Within_15_Cycles_Pct | 40.000000 |
| Mid_RUL_Samples | 60.000000 |
| End_of_Life_MAE | 7.960272 |
| End_of_Life_RMSE | 10.622386 |
| End_of_Life_R2 | -2.003390 |
| End_of_Life_Bias | 6.493126 |
| End_of_Life_Median_Absolute_Error | 6.846664 |
| End_of_Life_Dangerous_Overestimation_Pct | 83.018868 |
| End_of_Life_Underestimation_Pct | 16.981132 |
| End_of_Life_Within_5_Cycles_Pct | 43.396226 |
| End_of_Life_Within_10_Cycles_Pct | 67.924528 |
| End_of_Life_Within_15_Cycles_Pct | 84.905660 |
| End_of_Life_Samples | 53.000000 |
| Validation_Threshold_Neural_Accuracy | 0.895161 |
| Validation_Threshold_Neural_Precision | 0.690141 |
| Validation_Threshold_Neural_Recall | 0.924528 |
| Validation_Threshold_Neural_F1 | 0.790323 |
| Validation_Threshold_Neural_TN | 173.000000 |
| Validation_Threshold_Neural_FP | 22.000000 |
| Validation_Threshold_Neural_FN | 4.000000 |
| Validation_Threshold_Neural_TP | 49.000000 |
| Validation_Threshold_Neural_ROC_AUC | 0.976488 |
| Validation_Threshold_Neural_Average_Precision | 0.919678 |
| Safety_Accuracy | 0.895161 |
| Safety_Precision | 0.670886 |
| Safety_Recall | 1.000000 |
| Safety_F1 | 0.803030 |
| Safety_TN | 169.000000 |
| Safety_FP | 26.000000 |
| Safety_FN | 0.000000 |
| Safety_TP | 53.000000 |
| Safety_ROC_AUC | 0.978326 |
| Safety_Average_Precision | 0.925078 |
| Hybrid_Accuracy | 0.879032 |
| Hybrid_Precision | 0.638554 |
| Hybrid_Recall | 1.000000 |
| Hybrid_F1 | 0.779412 |
| Hybrid_TN | 165.000000 |
| Hybrid_FP | 30.000000 |
| Hybrid_FN | 0.000000 |
| Hybrid_TP | 53.000000 |
| Hybrid_ROC_AUC | 0.977455 |
| Hybrid_Average_Precision | 0.921078 |
| Posthoc_Neural_Accuracy | 0.931452 |
| Posthoc_Neural_Precision | 0.821429 |
| Posthoc_Neural_Recall | 0.867925 |
| Posthoc_Neural_F1 | 0.844037 |
| Posthoc_Neural_TN | 185.000000 |
| Posthoc_Neural_FP | 10.000000 |
| Posthoc_Neural_FN | 7.000000 |
| Posthoc_Neural_TP | 46.000000 |
| Posthoc_Neural_ROC_AUC | 0.976488 |
| Posthoc_Neural_Average_Precision | 0.919678 |
| NASA_PHM_Score_Raw | 4317.084961 |
| NASA_PHM_Score_Safety_Adjusted | 5690.887207 |
| Safety_Margin | 15.000000 |
| Validation_Selected_Fail_Probability_Threshold | 0.010000 |
| Posthoc_Test_Optimized_Fail_Probability_Threshold | 0.240000 |

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
