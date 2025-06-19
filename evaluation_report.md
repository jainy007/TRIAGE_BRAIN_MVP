# Triage Brain Evaluation Report

## Individual Model Performance
### XGBoost
- Accuracy: 0.922
- Precision: 0.967
- Recall: 0.906
- F1-Score: 0.935

### CNN Attention
- Accuracy: 0.627
- Precision: 0.627
- Recall: 1.000
- F1-Score: 0.771

### Autoencoder
- Accuracy: 0.451
- Precision: 0.600
- Recall: 0.375
- F1-Score: 0.462

### SVM RBF
- Accuracy: 0.784
- Precision: 0.784
- Recall: 0.906
- F1-Score: 0.841

## Ensemble Performance
- Accuracy: 0.922
- Precision: 0.967
- Recall: 0.906
- F1-Score: 0.935

## Optimized Weights
| Model | Weight |
|-------|--------|
| XGBoost | 0.400 |
| CNN+Attention | 0.150 |
| Autoencoder | 0.100 |
| SVM RBF | 0.350 |
