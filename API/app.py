import uvicorn
import pandas as pd 
from pydantic import BaseModel
from fastapi import FastAPI
from typing import  Union
from joblib import load


pricing = pd.read_csv("https://getaround-deployment.s3.eu-west-3.amazonaws.com/get_around_pricing_project.csv")
#Load model and preprocessor
loaded_model = load('model_reg.joblib')
preprocessor = load('preprocessor.joblib')


app = FastAPI(
    title="Car Rental price API",
    description="""
    This is our Getaround Predictor.
    You can use it to predict the price of your car rental.

    #Endpoints

    * You must use '/'  'get' to have the page result.
    * You must use '/predict' for a post request of the machine learning model.
    """
)
@app.get("/")
async def root():
    message = """ Welcome to the Getaround API"""
    return message


class PredictionFeatures(BaseModel):
    model_key: str 
    mileage: Union[int, float]
    engine_power: Union[int, float]
    fuel: str 
    paint_color: str 
    car_type: str 
    private_parking_available: bool 
    has_gps: bool 
    has_air_conditioning: bool 
    automatic_car: bool 
    has_getaround_connect: bool 
    has_speed_regulator: bool 
    winter_tires: bool 



@app.post("/predict", tags=["Machine Learning"])
def predict(predictionsFeatures:PredictionFeatures):
    
    features = dict(predictionsFeatures)
    data = pd.DataFrame(columns=['model_key', 'mileage', 'engine_power', 'fuel', 'paint_color','car_type', 'private_parking_available', 'has_gps',
    'has_air_conditioning', 'automatic_car', 'has_getaround_connect','has_speed_regulator', 'winter_tires'])[0]
   

    input_pred = preprocessor.transform(data)
    pred = loaded_model.predict(input_pred)
    return {"prediction" : pred[0]}

    
if __name__ == "__app__":
    uvicorn.run(app, host = "0.0.0.0", port = 4000, debug=True, reload=True)  

    
    

