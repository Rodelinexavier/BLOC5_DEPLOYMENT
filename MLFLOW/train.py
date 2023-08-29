import argparse
import pandas as pd
import time
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import  StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.compose import ColumnTransformer
import os, joblib
from pickle import dump
import pickle


if __name__ == "__main__":

    # Set your experiment name
    EXPERIMENT_NAME = "getaround Pricing v5"

    # Set tracking URI to your Heroku application
    mlflow.set_tracking_uri("https://my-getaround-mlflow-rodelin-610c1fba4675.herokuapp.com/")

    # Set experiment's info 
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Get our experiment info
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    # Time execution
    start_time = time.time()

    # Call mlflow autolog
    mlflow.sklearn.autolog()

    with mlflow.start_run(experiment_id=experiment.experiment_id):

        # Import dataset
        DATA_URL = "https://getaround-deployment.s3.eu-west-3.amazonaws.com/get_around_pricing_project.csv"
        pricing=pd.read_csv(DATA_URL)
        pricing=pricing.iloc[: , 1:]

        # Separate target variable y from features X
        target_name = "rental_price_per_day"
        y = pricing.loc[:,target_name]
        X = pricing.drop(target_name,axis=1)

        # Automatically detect names of numeric/categorical columns
        numeric_features = [1,2]
        categorical_features = [0,3,4,8,6,7,8,9,10,11,12]


        # Dividing into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

        # Create pipeline for categorical features
        categorical_transformer = OneHotEncoder(drop='first',handle_unknown = 'ignore', sparse=False)

        # Create pipeline for numeric features
        numeric_transformer = StandardScaler()

        # Use ColumnTransformer to make a preprocessor object that describes all the treatments to be done
        preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),('cat', categorical_transformer, categorical_features)])

        # Create a model

        regressor = LinearRegression()
        
        
        model = Pipeline(steps=[("Preprocessing", preprocessor),
                                ("Regressor", regressor)])
        

        # Log experiment to MLFlow
        with mlflow.start_run(experiment_id = experiment.experiment_id):
                model.fit(X_train, y_train)
                predictions = model.predict(X_train)


        # Log model seperately to have more flexibility on setup 
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="getaround_pricing_predictions",
            registered_model_name="Linear_Regression",
            signature=infer_signature(X_train, predictions)
        )


        




        
        







    

