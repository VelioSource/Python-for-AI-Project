import pandas as pd
import matplotlib.pyplot as plt
import textwrap


results_table = pd.DataFrame({
    "Method": [
        "XGBoost",
        "XGBoost 2",
        "Neural Network",
        "Neural Network 2"
    ],

    "Pre-processing step": [
        "Scaled",
        "Non-scaled",
        "Scaled",
        "Non-scaled"
    ],

    "Data Split": [
        "Random 80/20 stratified split",
        "Random 80/20 stratified split",
        "Random 80/20 stratified split + 20% validation split inside training",
        "Random 80/20 stratified split + 20% validation split inside training"
    ],

    "Accuracy": [
        "0.7500",
        "0.7500",
        "0.7473",
        "0.7473"
    ],

    "F1-score": [
        "0.8235",
        "0.8235",
        "0.8194",
        "0.8151"
    ],

    "ROC-AUC": [
        "0.8163",
        "0.8163",
        "0.8059",
        "0.8072"
    ],

    "Training Time": [
        "1min 18sec",
        "0min 53sec",
        "3min 31sec",
        "3min 27sec"
    ],

    "Tuning Time": [
        "8min 23sec",
        "7min 57sec",
        "0min 0sec",
        "0min 0sec"
    ],

    "Best Hyperparameters": [
        "n_estimators=500, learning_rate=0.01, max_depth=4, subsample=0.7",
        "n_estimators=500, learning_rate=0.01, max_depth=4, subsample=0.7",
        "neurons=[128,64,32], dropout=[0.3,0.3,0.2], batch_size=32, learning_rate=default",
        "neurons=[128,64,32], dropout=[0.3,0.3,0.2], batch_size=32, learning_rate=default"
    ],

    "Hyperparameter Tuning": [
        "GridSearchCV, Stratified 5-fold CV",
        "GridSearchCV, Stratified 5-fold CV",
        "Manual search / small tuning loop",
        "Manual search / small tuning loop"
    ]
})



results_table.to_csv("ufc_model_comparison_table_final.csv", index=False)

print("Saved CSV: ufc_model_comparison_table_final.csv")