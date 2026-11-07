import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np



mlflow.set_tracking_uri("http://localhost:5000")  # Set the MLflow tracking URI 
data=pd.read_csv("/home/poridhian/code/WineQT.csv")

X= data.drop("quality",axis=1)
y= data["quality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
params = {
    "n_estimators": 100,
    "max_depth": 5,
    "random_state": 42
}

model = RandomForestRegressor(**params)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse=np.sqrt(mean_squared_error(y_test, y_pred))
with mlflow.start_run() as run: 
    mlflow.log_params(params)
    mlflow.log_metric("rmse", rmse)
    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(model, "model", signature=signature)
    run_id = run.info.run_id
    print(f"Run ID: {run_id}")


