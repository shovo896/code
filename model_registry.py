import mlflow 

mlflow.set_tracking_uri("http://localhost:5000")  # Set the tracking URI for MLflow
from sklearn.ensemble import RandomForestClassifier 
client = mlflow.tracking.MlflowClient()  # Create an MLflow client instance

rfc = RandomForestClassifier(n_estimators=100, random_state=42)  # Initialize a Random Forest Classifier
with mlflow.start_run(run_name="logging_forest") as run:
    logged_model = mlflow.sklearn.log_model(rfc, name="random_forest_model")  # Log the model to MLflow
    mlflow.log_param("n_estimators", 100)  # Log the number of estimators as a parameter
    mlflow.log_param("random_state", 42)  # Log the random state as a parameter


model_name = "random_forest_model"  
model_version = mlflow.register_model(logged_model.model_uri, model_name)
result = client.get_registered_model(model_name)

print(f"Registered Model Name: {result.name}")  # Print the name of the registered model
print(f"Registered Model Creation Time: {result.creation_timestamp}")  # Print the creation timestamp of the registered model
print(f"Registered Model Last Updated Time: {result.last_updated_timestamp}")  # Print the last updated timestamp of the registered model
print(f"Registered Model Description: {result.description}")  # Print the description of the registered model
print(f"Model Version: {model_version.version}")  # Print the version of the registered model
print(f"Model Version Stage: {model_version.current_stage}")  # Print the current stage of the registered model version
print()
