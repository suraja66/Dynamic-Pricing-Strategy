from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
from typing import Dict

app = FastAPI()

model_filename = 'Dynamic_Pricing_Strategy.pkl'

# Dummy Training for the Model (Replace with your actual training logic)
def train_and_save_model():
    # Dummy data for training the model
    data = pd.DataFrame({
        'Number_of_Riders': [50],
        'Number_of_Drivers': [25],
        'Vehicle_Type': [0],
        'Expected_Ride_Duration': [30]
    })

    X = data[['Number_of_Riders', 'Number_of_Drivers', 'Vehicle_Type', 'Expected_Ride_Duration']]
    y = data['Price']

    model = RandomForestClassifier()
    model.fit(X, y)

    # Save the trained model
    joblib.dump(model, model_filename)
    print("Model trained and saved successfully!")

# Load model during startup
@app.on_event("startup")
def load_model():
    global model
    try:
        model = joblib.load(model_filename)
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Model file not found. Training a new model...")
        train_and_save_model()
        model = joblib.load(model_filename)
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")

class PricingRequest(BaseModel):
    Number_of_Riders: float                      
    Number_of_Drivers: float 
    Vehicle_Type: float                    
    Expected_Ride_Duration: float                  

@app.get('/')
def home():
    return {'message': 'Dynamic Pricing API is running!'}

@app.get('/health')
def health_check():
    return {'status': 'Healthy'}

@app.post('/predict')
def predict_price(request: PricingRequest):
    try:
        # Prepare the input data for prediction
        input_data = [[
            request.Number_of_Riders, 
            request.Number_of_Drivers, 
            request.Vehicle_Type, 
            request.Expected_Ride_Duration
        ]]
        
        # Make prediction
        prediction = model.predict(input_data)
        price = prediction[0]

        return {'predicted_price': price}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
