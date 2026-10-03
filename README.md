# Supervised Learning Model Implementation - Breast Cancer Wisconsin (Diagnostic)

Week 4 task of the Yuva Intern *Data Science with Python* virtual internship.

## Goal
Build and evaluate a supervised classification model on a public dataset,
with proper train/test splitting, cross-validation, and multiple metrics.

## Dataset
Breast Cancer Wisconsin (Diagnostic), UCI ML Repository (569 samples, reused
from Weeks 2-3). Loaded via `sklearn.datasets.load_breast_cancer`.
Target: `is_malignant` (1 = malignant, 0 = benign).

## What the script does
1. Stratified 80/20 train-test split
2. Scales features for Logistic Regression (fit on train only)
3. Trains Logistic Regression (baseline) and a GridSearchCV-tuned Random Forest
4. 5-fold cross-validation for both models
5. Evaluates on the held-out test set: accuracy, precision, recall, F1, ROC-AUC
6. Plots confusion matrices, ROC curves, CV score spread, and feature importance

## Run
```
pip install -r requirements.txt
python supervised_breast_cancer.py
```

## Key results (held-out test set, 114 samples)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.965 | 0.975 | 0.929 | 0.951 | 0.996 |
| Random Forest | 0.974 | 1.000 | 0.929 | 0.963 | 0.995 |

Top predictive features: worst area, worst concave points, worst radius.

Full write-up: `Week4_Supervised_Learning_Report.docx`
