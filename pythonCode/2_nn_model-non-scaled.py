# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix,roc_auc_score, RocCurveDisplay

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# %%
#Loading data
df = pd.read_csv(r"C:\Users\Velimir\iCloudDrive\Documents\Python for AI\Project\Large set\ufc_processed.csv")

# %%
#Adding features and scaler
feature_names = joblib.load(r"C:\Users\Velimir\iCloudDrive\Documents\Python for AI\ufc_features.pkl")
scaler=joblib.load(r"C:\Users\Velimir\iCloudDrive\Documents\Python for AI\ufc_scaler.pkl")

print("Feature names loaded.")
print("Scaler loaded.")
print("Number of features:", len(feature_names))

# %%
X = df.drop(columns=["winner"])
y = df["winner"]
#making sure x has the same order as the pkl file
X = X[feature_names]

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
#Sequential NN model creation

'''

128 neurons  ->  64 neurons  ->  32 neurons  ->  1 neuron

'''

model = Sequential()

model.add(Dense(128, activation="relu", input_shape=(X_train.shape[1],)))
'''
It works by normalizing the data within each mini-batch. This means it calculates the mean
and variance of data in a batch and then adjusts the values so that they have similar range. 
After that it scales and shifts the values so that model learn effectively.
'''
model.add(BatchNormalization())
'''
During training, randomly switches off 30% of neurons each pass. 
Forces the network to not rely too heavily on any single neuron — reduces overfitting.
'''
model.add(Dropout(0.3))

model.add(Dense(64, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(1, activation="sigmoid"))

# %%
#Compiling model

model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["accuracy"]
)

model.summary()


# %%
#Adding Callbacks

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "best_ufc_nn_model_non_scaled.keras",
    monitor="val_loss",
    save_best_only=True
)


# %%
#Training the NN model



history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, checkpoint],
    verbose=1
)


# %%
#Predictions
y_probability_nn = model.predict(X_test)

y_pred_nn = np.round(y_probability_nn)
y_pred_nn = y_pred_nn.flatten().astype(int)

# %%
#Checking accuracy
accuracy_nn = accuracy_score(y_test, y_pred_nn)
f1_nn = f1_score(y_test, y_pred_nn)

print("Neural Network Accuracy:", accuracy_nn)
print("Neural Network F1-score:", f1_nn)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_nn))



# %%
# 11. EVALUATE

accuracy_nn = accuracy_score(y_test, y_pred_nn)
f1_nn = f1_score(y_test, y_pred_nn)

print("Neural Network Accuracy:", accuracy_nn)
print("Neural Network F1-score:", f1_nn)

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

model.save("ufc_neural_network_non_scaled.keras")

print("Neural Network model saved as ufc_neural_network_non_scaled.keras")