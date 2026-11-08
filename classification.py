import mlflow 
import xgboost as xgb
import shap 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature

mlflow.set_tracking_uri("http://localhost:5000") 
experiment_name = "XGBoost_Classification" 
try:
    mlflow.create_experiment(experiment_name)
except Exception as e:
    print(f"Experiment '{experiment_name}' already exists. Using the existing experiment.")

experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

X,y = shap.datasets.adult()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)
eval_data= X_test.copy()
eval_data["label"] = y_test
predictions = model.predict(X_test)


with mlflow.start_run(experiment_id=experiment_id):
    signature = infer_signature(X_train, model.predict(X_train)) 
    mlflow.xgboost.log_model(model, "xgboost_model", signature=signature)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)
    mlflow.log_metric("accuracy", accuracy)
    print("Evaluation results:") 
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{matrix}")
    print(f"Classification Report:\n{report}")
