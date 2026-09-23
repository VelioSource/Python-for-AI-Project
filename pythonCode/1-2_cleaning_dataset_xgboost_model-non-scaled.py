# %%
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, RocCurveDisplay
)
import matplotlib.pyplot as plt

# %%
# Load data
df = pd.read_csv(r"C:\Users\Velimir\iCloudDrive\Documents\Python for AI\Project\Large set\ufc_processed.csv")

assert df.isnull().sum().sum() == 0, "Dataset contains missing values!"
assert all(df.dtypes != "object"), "Dataset contains non-numerical columns!"
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# %%
# Features and target
X = df.drop(columns=["winner"])
y = df["winner"]

feature_names = X.columns.tolist()

print(f"Features: {X.shape[1]}")
print(f"Class balance:\n{y.value_counts(normalize=True).round(3)}")

# %%
# Train / Test split (80/20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# %%
# GridSearchCV — finds best hyperparameters using 5 folds
#
# No separate validation set needed because CV handles it

param_grid = {
    "n_estimators":  [400, 500, 700, 800],
    "learning_rate": [0.001,0.01, 0.05],
    "max_depth":     [4, 5,6,7],
    "subsample":     [0.6,0.7,0.8]
}

base_model = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=cv,
    scoring="f1",
    n_jobs=-1,        # use all CPU cores
    verbose=1,
    refit=True        # automatically retrains best model on full train set
)

print("Running GridSearchCV... (this may take a few minutes)")
grid_search.fit(X_train, y_train)

print(f"\nBest parameters:  {grid_search.best_params_}")
print(f"Best CV F1-score: {grid_search.best_score_:.4f}")

# %%
# Best model is already refitted on full train set by GridSearchCV (refit=True)
xgb_model = grid_search.best_estimator_

# %%
# Evaluation on test set
y_pred  = xgb_model.predict(X_test)
y_proba = xgb_model.predict_proba(X_test)[:, 1]

print("=== Test Set ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Blue", "Red"]))

# %%
# Confusion matrix
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["Blue", "Red"],
    cmap="Blues",
    ax=ax
)
ax.set_title("XGBoost — Confusion Matrix")
plt.tight_layout()
plt.show()

# %%
# ROC curve
fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax, name="XGBoost")
ax.plot([0, 1], [0, 1], "k--", label="Random")
ax.set_title("XGBoost — ROC Curve")
ax.legend()
plt.tight_layout()
plt.show()

# %%
# Feature importances
feature_importance_df = pd.DataFrame({
    "Feature":    feature_names,
    "Importance": xgb_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTop 15 Most Important Features:")
print(feature_importance_df.head(15).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 6))
top15 = feature_importance_df.head(15)
ax.barh(top15["Feature"][::-1], top15["Importance"][::-1])
ax.set_title("Top 15 XGBoost Feature Importances")
ax.set_xlabel("Importance")
ax.set_ylabel("Feature")
plt.tight_layout()
plt.show()

# %%
# Save model
joblib.dump(xgb_model, "ufc_xgboost_model_non_scaled.pkl")
print("Model saved as ufc_xgboost_model_non_scaled.pkl")