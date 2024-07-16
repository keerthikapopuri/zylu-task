from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load the trained models
model_return = joblib.load('models/model_return.pkl')
model_repurchase = joblib.load('models/model_repurchase.pkl')
scaler = joblib.load('models/scaler.pkl')

class PredictionRequest(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    # Add other features as necessary

@app.post("/predict_return")
def predict_return(request: PredictionRequest):
    data = np.array([[
        request.feature1,
        request.feature2,
        request.feature3,
        # Add other features in the same order as your model expects
    ]])
    
    data = scaler.transform(data)
    prediction = model_return.predict(data)
    return {"return_likelihood": int(prediction[0])}

@app.post("/predict_repurchase")
def predict_repurchase(request: PredictionRequest):
    data = np.array([[
        request.feature1,
        request.feature2,
        request.feature3,
        # Add other features in the same order as your model expects
    ]])
    
    data = scaler.transform(data)
    prediction = model_repurchase.predict(data)
    return {"repurchase_likelihood": int(prediction[0])}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
