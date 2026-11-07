from fastapi import FastAPI
from pydantic import BaseModel 
import uvicorn 
import os 
import mlflow 
import mlflow.sklearn 
import pandas as pd
import numpy as np 

app= FastAPI() 
mlflow.set_tracking_uri("http://localhost:5000")  # Set the MLflow tracking URI 

model_uri ="models:/RandomForestWineQuality/1"  # Replace with your model's URI
model = mlflow.sklearn.load_model(model_uri)
try : 
    model = mlflow.sklearn.load_model(model_uri)
    print("Model loaded successfully.") 
    print(f"Model URI: {model_uri}")
    if hasattr(model,'feature_names_in_'):
        print(f"Feature names: {model.feature_names_in_}")
except Exception as e:
    print(f"Error loading model: {e}")


###define the input data scema 
class WineQualityInput(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

 @app.get("/")
def read_root():
    return {"message": "Welcome to the Wine Quality Prediction API!"}


@app.get("/test")
def test_endpoint():
    return {"message": "The API is working!"}

     try: 
        test_data = pd.DataFrame([{
            "fixed_acidity": 7.4,
            "volatile_acidity": 0.7,
            "citric_acid": 0.0,
            "residual_sugar": 1.9,
            "chlorides": 0.076, 
            "free_sulfur_dioxide": 11.0,
            "total_sulfur_dioxide": 34.0,
            "density": 0.9978,
            "pH": 3.51,
            "sulphates": 0.56,
            "alcohol": 9.4
        }])
        prediction = model.predict(test_data)
        print(f"Test prediction: {prediction}")
    except Exception as e:
        print(f"Error during test prediction: {e}")
@app.post("/predict")
def predict_quality(input_data: WineQualityInput):
    try:
        # Test with exact column names the model expects
        test_data = pd.DataFrame([{
            "fixed acidity": 7.4,
            "volatile acidity": 0.7,
            "citric acid": 0.0,
            "residual sugar": 1.9,
            "chlorides": 0.076,
            "free sulfur dioxide": 11.0,
            "total sulfur dioxide": 34.0,
            "density": 0.9978,
            "pH": 3.51,
            "sulphates": 0.56,
            "alcohol": 9.4
        }])
        
        print(f"Test DataFrame columns: {list(test_data.columns)}")
        print(f"Test DataFrame:\n{test_data}")
        
        # Try prediction
        prediction = model.predict(test_data)
        return {"test_prediction": float(prediction[0]), "status": "success"}
        
    except Exception as e:
        print(f"Test prediction failed: {str(e)}")
        return {"error": str(e), "status": "failed"}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
