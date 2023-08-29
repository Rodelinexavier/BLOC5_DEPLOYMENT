import uvicorn
import mlflow 
import uvicorn
import pandas as pd 
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import RedirectResponse



mlflow.set_tracking_uri("https://getaround-mlflow-projet-5c161c389b6d.herokuapp.com/")

description = """ 

This is our Getaround Predictor.
You can use it to predict the price of your car rental.

#Endpoints

* You must use '/'  'get' to have the page result.
* You must use '/predict' for a post request of the machine learning model.
"""

app = FastAPI(
    title="Car Rental price API",
    description=description,
    version="0.1"
)


class PredictionFeatures(BaseModel):
    model_key: str = "Peugeat"
    mileage: int = 5000
    engine_power: int = 150
    fuel: str = "diesel"
    paint_color: str = "red"
    car_type: str = "coupe"
    private_parking_available: bool = True
    has_gps: bool = False
    has_air_conditioning: bool = True
    automatic_car: bool = True
    has_getaround_connect: bool = False
    has_speed_regulator: bool = True
    winter_tires: bool = False


@app.get("/", tags=["Endpoints"])
async def index():

    message = "Welcome to the getaound API"

    return message

@app.post("/predict", tags=["Machine Learning"])
async def predict(predictionFeatures: PredictionFeatures):

    # Read data 
    df_vehicle = pd.DataFrame(dict(predictionFeatures), index=[0])

    # Log model from mlflow 
    logged_model = 'runs:/45a2b3b50dc046aebdb46107115f17e1/getaround_prediction'
    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    # Predict on a Pandas DataFrame.
    import pandas as pd
    loaded_model.predict(pd.DataFrame(df_vehicle))
    prediction = loaded_model.predict(df_vehicle)


    # Format response
    response = {"prediction": prediction.tolist()[0]}
    return response

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000)

    
    

