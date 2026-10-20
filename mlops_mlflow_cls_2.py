import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("WineQT.csv")
X = data.drop("quality", axis=1)
y = (data["quality"] >= 7).astype(int)  # Binary classification: quality >= 7 is "good"
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set experiment
mlflow.set_experiment("Wine_Quality_Classification")

# Model 1: Random Forest
with mlflow.start_run(run_name="RandomForest_Trial"):
    rf_params = {"n_estimators": 100, "max_depth": 5}
    rf_model = RandomForestClassifier(**rf_params, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)

    # Log parameters and metrics
    mlflow.log_params(rf_params)
    mlflow.log_metric("accuracy", rf_accuracy)

    # Log confusion matrix as artifact
    cm = confusion_matrix(y_test, rf_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Random Forest Confusion Matrix")
    plt.savefig("rf_confusion_matrix.png")
    mlflow.log_artifact("rf_confusion_matrix.png")

    # Log model
    mlflow.sklearn.log_model(
        rf_model,
        name="random_forest_model",
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )
    print(f"Random Forest Accuracy: {rf_accuracy}")

# Model 2: Logistic Regression
with mlflow.start_run(run_name="LogisticRegression_Trial"):
    lr_params = {"C": 1.0, "max_iter": 1000}
    lr_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(**lr_params, random_state=42),
    )
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_pred)

    # Log parameters and metrics
    mlflow.log_params(lr_params)
    mlflow.log_metric("accuracy", lr_accuracy)

    # Log confusion matrix as artifact
    cm = confusion_matrix(y_test, lr_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Logistic Regression Confusion Matrix")
    plt.savefig("lr_confusion_matrix.png")
    mlflow.log_artifact("lr_confusion_matrix.png")

    # Log model
    mlflow.sklearn.log_model(lr_model, name="logistic_regression_model")
    print(f"Logistic Regression Accuracy: {lr_accuracy}")

        
