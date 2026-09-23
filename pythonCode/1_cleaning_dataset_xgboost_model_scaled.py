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
df = pd.read_csv(r"C:\Users\Velimir\iCloudDrive\Documents\Python for AI\Velimir_Velikov_UFC_Predition_Project\Large set\ufc_processed.csv")

assert df.isnull().sum().sum() == 0, "Dataset contains missing values!"
assert all(df.dtypes != "object"), "Dataset contains non-numerical columns!"
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# %%
# Features and target
X = df.drop(columns=["winner"])
y = df["winner"]
#before pandas index object
#used because of saving the model so every column is the exact same value each time
feature_names = X.columns.tolist()
#shape reutnr rows and columns, in this case returns columns
print(f"Features: {X.shape[1]}")
#.value_counts() counts how many times each value appears, normalize=True converts raw counts to proportions
print(f"Class balance:\n{y.value_counts(normalize=True).round(3)}")

# %%
# Train / Test split (80/20, stratified)
'''
stratify=y
Without it, the random split might 
accidentally put most "Red wins" in train and most "Blue wins" in test.
stratify=y guarantees both splits have the same class ratio as the original data.
'''
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# %%
# Feature scaling — fit on train only
scaler = StandardScaler()
#learn standart devation and mean from train data
X_train_scaled = scaler.fit_transform(X_train)
#apply the same calculations on test data
X_test_scaled  = scaler.transform(X_test)

joblib.dump(scaler, "ufc_scaler.pkl")
joblib.dump(feature_names, "ufc_features.pkl")
print("Scaler and feature names saved.")

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


'''
 defines how the model measures its own error during training. 
 Logloss works well for binary classification (Red/Blue) 
it penalizes confident wrong predictions heavily:

'''
base_model = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)


'''
This defines how the cross-validation splits work.
n_splits=5 — splits training data into 5 chunks:
Fold 1: [TEST][train][train][train][train]
Fold 2: [train][TEST][train][train][train]
Fold 3: [train][train][TEST][train][train]
...
Each fold trains on 4 chunks, tests on 1. This repeats 5 times so every row gets tested exactly once.
shuffle=True — randomly shuffles data before splitting, so folds aren't accidentally ordered (e.g. all early fights in fold 1).
Stratified — same idea as before, each fold maintains the same Red/Blue class ratio.



'''
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=base_model,# the model to tune
    param_grid=param_grid, # all combinations to try
    cv=cv, # use the 5-fold strategy above
    scoring="f1", # judge each combo by F1 score
    n_jobs=-1,        # use all CPU cores
    verbose=1,       # print progress while running
    refit=True        # automatically retrains best model on full train set
)

print("Running GridSearchCV... (this may take a few minutes)")
grid_search.fit(X_train_scaled, y_train)

print(f"\nBest parameters:  {grid_search.best_params_}")
print(f"Best CV F1-score: {grid_search.best_score_:.4f}")

# %%
# Best model is already refitted on full train set by GridSearchCV (refit=True)
xgb_model = grid_search.best_estimator_

# %%
# Evaluation on test set
y_pred  = xgb_model.predict(X_test_scaled)

'''
xgb_model.predict_proba(X_test_scaled) — returns probabilities for both classes for each fight:
        Blue    Red
row 0: [0.30,  0.70]
row 1: [0.80,  0.20]
row 2: [0.45,  0.55]
It always returns two columns — probability of 0 (Blue) and probability of 1 (Red). They always add up to 1.
'''

'''
The Slicing [:, 1]
python[:, 1]
# :  → all rows
# 1  → column index 1 (the Red probability)
So you're just grabbing the Red win probability column:
[0.70, 0.20, 0.55, ...]
'''
y_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]

print("=== Test Set ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Blue", "Red"]))

# %%
# Confusion matrix


'''
fig, ax = plt.subplots(figsize=(5, 4))
Creates a blank figure. fig is the whole window, ax is the plot area inside it. figsize=(5,4) is width x height in inches.
pythonConfusionMatrixDisplay.from_predictions(y_test, y_pred, ...)
Takes real labels vs predicted labels and builds a grid showing:
                Predicted Blue   Predicted Red
Actual Blue  [      TN        |      FP      ]
Actual Red   [      FN        |      TP      ]
Where each cell counts how many fights landed there — lets you see not just accuracy but what kind of mistakes the model makes.
'''
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
'''
"k"  → black color  (k = black in matplotlib)
"--" → dashed line
'''
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
#horizontal bar chart
ax.barh(top15["Feature"][::-1], top15["Importance"][::-1])
ax.set_title("Top 15 XGBoost Feature Importances")
ax.set_xlabel("Importance")
ax.set_ylabel("Feature")
#tight_layout() automatically adjusts spacing so labels don't get cut off. 
plt.tight_layout()
plt.show()

# %%
# Save model
joblib.dump(xgb_model, "ufc_xgboost_model.pkl")
print("Model saved as ufc_xgboost_model.pkl")