
from xml.parsers.expat import model

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from schema.user_input import UserInput
from schema.prediction_response import PredictionResponse
import pickle
import pandas as pd

from model.predict import predict_output, MODEL_VERSION, model


app= FastAPI()

    
@app.get('/')
def home():
    return {"message": "Welcome to the Insurance Premium Predictor API. Use the /predict endpoint to get a prediction."}

@app.get('/health')
def health_check():
    return {"status": "ok", "version": MODEL_VERSION, 'model_loaded': model is not None}


@app.post('/predict',response_model=PredictionResponse)
def predict_premium(data: UserInput):
    user_input = {
        'bmi': data.bmi,
        'age_group': data.age_group,
        'lifestyle_risk': data.lifestyle_risk,
        'city_tier': data.city_tier,
        'income_lpa': data.income_lpa,
        'occupation': data.occupation
    }

    try:
        prediction= predict_output(user_input)

        return JSONResponse(status_code=200, content={'response ': prediction})
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})
    
    


