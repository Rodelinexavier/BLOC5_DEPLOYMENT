docker run -it\
-p 4000:4000\
-v "$(pwd):/home/app"\
-e MLFLOW_TRACKING_URI="https://my-getaround-mlflow-rodelin-610c1fba4675.herokuapp.com/"\ 
-e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
-e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY\
-e ARTIFACT_STORE_URI=$ARTIFACT_STORE_URI\
-e BACKEND_STORE_URI=$BACKEND_STORE_URI \
my-mlflow-image python train.py