"""
Week 4 - Supervised Learning Model Implementation
Dataset: Breast Cancer Wisconsin (Diagnostic) - UCI ML Repository
Task: Binary classification (malignant vs benign)
Libraries: Pandas, Scikit-learn, Matplotlib, Seaborn
Run: python supervised_breast_cancer.py
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix, roc_curve, roc_auc_score, ConfusionMatrixDisplay)

sns.set_theme(style="whitegrid", palette="Set2")
FIG = "figures/"
RANDOM_STATE = 42

# ------------------------------------------------------------------
# 1. Load data and define the problem
# ------------------------------------------------------------------
raw = load_breast_cancer(as_frame=True)
df = raw.frame.copy()
df["diagnosis"] = df["target"].map({0: "Malignant", 1: "Benign"})
# Classification target: 1 = malignant (the positive/important class to catch)
df["is_malignant"] = (df["diagnosis"] == "Malignant").astype(int)
feature_cols = [c for c in df.columns if c not in ("target", "diagnosis", "is_malignant")]
print("Features:", len(feature_cols))
print(df["is_malignant"].value_counts())

X = df[feature_cols]
y = df["is_malignant"]

# ------------------------------------------------------------------
# 2. Train / test split (stratified)
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
print("Train size:", X_train.shape, "Test size:", X_test.shape)
print("Train malignant rate:", y_train.mean().round(3), "Test malignant rate:", y_test.mean().round(3))

# ------------------------------------------------------------------
# 3. Preprocessing: scale features (fit on train only)
# ------------------------------------------------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ------------------------------------------------------------------
# 4. Feature correlation with target (quick feature engineering check)
# ------------------------------------------------------------------
corr_target = X_train.assign(y=y_train.values).corr()["y"].drop("y").sort_values(key=abs, ascending=False)
print(corr_target.head(10))

# ------------------------------------------------------------------
# 5. Baseline model: Logistic Regression
# ------------------------------------------------------------------
logreg = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores_lr = cross_val_score(logreg, X_train_s, y_train, cv=cv, scoring="accuracy")
print("LogReg 5-fold CV accuracy: %.3f +/- %.3f" % (cv_scores_lr.mean(), cv_scores_lr.std()))

logreg.fit(X_train_s, y_train)
pred_lr = logreg.predict(X_test_s)
proba_lr = logreg.predict_proba(X_test_s)[:, 1]

# ------------------------------------------------------------------
# 6. Second model: Random Forest + small grid search
# ------------------------------------------------------------------
rf = RandomForestClassifier(random_state=RANDOM_STATE)
param_grid = {"n_estimators": [100, 200], "max_depth": [None, 5, 10]}
grid = GridSearchCV(rf, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)  # tree model: raw features, no scaling needed
print("Best RF params:", grid.best_params_, "CV accuracy: %.3f" % grid.best_score_)
best_rf = grid.best_estimator_
pred_rf = best_rf.predict(X_test)
proba_rf = best_rf.predict_proba(X_test)[:, 1]

# ------------------------------------------------------------------
# 7. Evaluation on the held-out test set
# ------------------------------------------------------------------
def report(name, y_true, y_pred, y_proba):
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }

results = pd.DataFrame([
    report("Logistic Regression", y_test, pred_lr, proba_lr),
    report("Random Forest", y_test, pred_rf, proba_rf),
]).round(3)
print(results)
results.to_csv("model_results.csv", index=False)

# ------------------------------------------------------------------
# 8. Confusion matrices
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
ConfusionMatrixDisplay(confusion_matrix(y_test, pred_lr), display_labels=["Benign", "Malignant"]).plot(ax=axes[0], cmap="Greens", colorbar=False)
axes[0].set_title("Figure 1a: Logistic Regression")
ConfusionMatrixDisplay(confusion_matrix(y_test, pred_rf), display_labels=["Benign", "Malignant"]).plot(ax=axes[1], cmap="Oranges", colorbar=False)
axes[1].set_title("Figure 1b: Random Forest")
plt.tight_layout(); plt.savefig(FIG + "fig1_confusion_matrices.png", dpi=150); plt.close()

# ------------------------------------------------------------------
# 9. ROC curves
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
for name, proba in [("Logistic Regression", proba_lr), ("Random Forest", proba_rf)]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
ax.set_title("Figure 2: ROC curves")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend()
plt.tight_layout(); plt.savefig(FIG + "fig2_roc_curves.png", dpi=150); plt.close()

# ------------------------------------------------------------------
# 10. Cross-validation stability
# ------------------------------------------------------------------
cv_scores_rf = cross_val_score(best_rf, X_train, y_train, cv=cv, scoring="accuracy")
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.boxplot([cv_scores_lr, cv_scores_rf], labels=["Logistic\nRegression", "Random\nForest"])
ax.set_title("Figure 3: 5-fold cross-validation accuracy")
ax.set_ylabel("Accuracy")
plt.tight_layout(); plt.savefig(FIG + "fig3_cv_boxplot.png", dpi=150); plt.close()
print("RF 5-fold CV accuracy: %.3f +/- %.3f" % (cv_scores_rf.mean(), cv_scores_rf.std()))

# ------------------------------------------------------------------
# 11. Feature importance (Random Forest) and coefficients (LogReg)
# ------------------------------------------------------------------
importances = pd.Series(best_rf.feature_importances_, index=feature_cols).sort_values(ascending=False).head(10)
fig, ax = plt.subplots(figsize=(7, 5))
importances.sort_values().plot(kind="barh", ax=ax, color="#fc8d62")
ax.set_title("Figure 4: Top 10 Random Forest feature importances")
ax.set_xlabel("Importance")
plt.tight_layout(); plt.savefig(FIG + "fig4_feature_importance.png", dpi=150); plt.close()
print(importances)

coefs = pd.Series(logreg.coef_[0], index=feature_cols).sort_values(key=abs, ascending=False).head(10)
print(coefs)

print("\nDone.")
