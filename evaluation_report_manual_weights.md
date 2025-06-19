# Triage Brain Evaluation Report

## Individual Model Performance
### Xgboost
- Accuracy: 0.922
- Precision: 0.967
- Recall: 0.906
- F1-Score: 0.935

### Cnn Attention
- Accuracy: 0.627
- Precision: 0.627
- Recall: 1.000
- F1-Score: 0.771

### Autoencoder
- Accuracy: 0.392
- Precision: 0.600
- Recall: 0.094
- F1-Score: 0.162

### Svm Rbf
- Accuracy: 0.784
- Precision: 0.784
- Recall: 0.906
- F1-Score: 0.841

## Ensemble Performance
- Accuracy: 0.941
- Precision: 0.968
- Recall: 0.938
- F1-Score: 0.952

## Optimized Weights
| Model | Weight |
|-------|--------|
| XGBoost | 0.400 |
| CNN+Attention | 0.150 |
| AutoEncoder | 0.100 |
| SVM RBF | 0.350 |
