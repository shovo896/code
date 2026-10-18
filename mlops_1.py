mport mlflow 
from sklearn.datasets import load_diabetes 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error 
import numpy as np


mlflow.set_experiment("Diabetes_Regression_Experiment")
diabetes = load_diabetes() 
X,y = diabetes.data, diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

learning_rate = 0.01 
n_iterations = 1000 


## start a run 
with mlflow.start_run(run_name = "SGD_regression_trail"): 
    model = SGDRegressor(learning_rate='constant', eta0=learning_rate, max_iter=n_iterations, random_state=42) 
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("n_iterations", n_iterations)
    mlflow.log_metric("mse", mse)

    with open("model_summary.txt", "w") as f: 
        f.write(f"Model Coefficients: {model.coef_}\n")
        f.write(f"Model Intercept: {model.intercept_}\n")

        mlflow.log_artifact("model_summary.txt")
        print(f"Run completed with MSE : {mse}")
