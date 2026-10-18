import mlflow 
from sklearn.datasets import load_diabetes 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error 
import numpy as np


mlflow.set_experiment("Diabetes_Regression_Experiment") 
diabetes = load_diabetes() 
X,y = diabetes.data, diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
max_depth = 5 
max_samples_split = 4 
random_state = 42 
with  mlflow.start_run(run_name = "Decision_tree_regressor_trail"): 
    model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=max_samples_split, random_state=random_state) 
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("max_samples_split", max_samples_split)
    mlflow.log_metric("mse", mse)

    with open("model_summary.txt", "w") as f: 
        f.write(f"Model Feature Importances: {model.feature_importances_}\n")
        f.write(f"Model Depth: {model.get_depth()}\n")

        mlflow.log_artifact("model_summary.txt")
        print(f"Run completed with MSE : {mse}")
