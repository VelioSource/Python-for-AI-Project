# %%
# Import libraries

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# %%
# Load the processed UFC dataset

df = pd.read_csv(
    r"C:\Users\Velimir\iCloudDrive\Documents\Python for AI\Project\Large set\ufc_processed.csv"
)

print("Dataset loaded.")
print("Dataset shape:", df.shape)


# %%
# Load feature names and scaler

feature_names = joblib.load(
    r"C:\Users\Velimir\iCloudDrive\Documents\Python for AI\ufc_features.pkl"
)

scaler = joblib.load(
    r"C:\Users\Velimir\iCloudDrive\Documents\Python for AI\ufc_scaler.pkl"
)

print("Feature names loaded.")
print("Scaler loaded.")
print("Number of features:", len(feature_names))


# %%
# Check dataset

print("Missing values:", df.isnull().sum().sum())
print("Non-numerical columns:", df.select_dtypes(include=["object"]).columns.tolist())


# %%
# Split data into features X and target y

X = df.drop(columns=["winner"])
y = df["winner"]

# Make sure the columns are in the same order as when the scaler was created
X = X[feature_names]

print("X shape:", X.shape)
print("y shape:", y.shape)


# %%
# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# %%
# Scale the data using the saved scaler

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaled data ready.")
print("X_train_scaled:", X_train_scaled.shape)
print("X_test_scaled:", X_test_scaled.shape)


# %%
# Create the Neural Network model

model = Sequential()

# First hidden layer
model.add(Dense(128, activation="relu", input_shape=(X_train_scaled.shape[1],)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Second hidden layer
model.add(Dense(64, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Third hidden layer
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))

# Output layer
# Sigmoid is used because this is a binary classification problem, sigmoid was chosen with ai
model.add(Dense(1, activation="sigmoid"))


# %%
# Compile the model

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# %%
# Create callbacks

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "best_ufc_nn_model.keras",
    monitor="val_loss",
    save_best_only=True
)


# %%
# Train the model

history = model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, checkpoint],
    verbose=1
)


# %%
# Make predictions

# The neural network outputs probabilities between 0 and 1
y_probability_nn = model.predict(X_test_scaled).flatten()

# Convert probabilities into class predictions
# 0 = Blue wins
# 1 = Red wins
y_pred_nn = (y_probability_nn >= 0.5).astype(int)


# %%
# Evaluate the model

accuracy_nn = accuracy_score(y_test, y_pred_nn)
f1_nn = f1_score(y_test, y_pred_nn)
roc_auc_nn = roc_auc_score(y_test, y_probability_nn)

print("Neural Network Accuracy:", accuracy_nn)
print("Neural Network F1-score:", f1_nn)
print("Neural Network ROC-AUC:", roc_auc_nn)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_nn))


# %%
# Confusion Matrix, initially the plot was very basic, so I used AI to make it look better

cm = confusion_matrix(y_test, y_pred_nn)

plt.figure(figsize=(5, 4))
plt.imshow(cm)
plt.title("Neural Network Confusion Matrix")
plt.colorbar()

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks([0, 1], ["Blue", "Red"])
plt.yticks([0, 1], ["Blue", "Red"])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.show()


# %%
# ROC Curve

fig, ax = plt.subplots(figsize=(6, 5))

RocCurveDisplay.from_predictions(
    y_test,
    y_probability_nn,
    ax=ax,
    name="Neural Network"
)

ax.plot([0, 1], [0, 1], "k--", label="Random Guessing")
ax.set_title("Neural Network ROC Curve")
ax.legend()

plt.tight_layout()
plt.show()


# %%
# Training and validation loss plot

plt.figure(figsize=(7, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Neural Network Loss During Training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()


# %%
# Training and validation accuracy plot

plt.figure(figsize=(7, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Neural Network Accuracy During Training")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()


# %%
# Save final neural network model

model.save("ufc_v1_neural_network.keras")

print("Neural Network model saved as ufc_v1_neural_network.keras")