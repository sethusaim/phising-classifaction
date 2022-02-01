import json
import os

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates

from phising.model.load_production_model import Load_Prod_Model
from phising.model.prediction_from_model import Prediction
from phising.model.training_model import Train_Model
from phising.validation_insertion.prediction_validation_insertion import Pred_Validation
from phising.validation_insertion.train_validation_insertion import Train_Validation
from utils.read_params import read_params

os.putenv("LANG", "en_US.UTF-8")
os.putenv("LC_ALL", "en_US.UTF-8")

app = FastAPI()

config = read_params()

templates = Jinja2Templates(directory=config["templates"]["dir"])

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(
        config["templates"]["index_html_file"], {"request": request}
    )


@app.get("/train")
async def trainRouteClient():
    try:
        raw_data_train_bucket_name = config["s3_bucket"]["phising_raw_data_bucket"]

        train_valObj = Train_Validation(raw_data_train_bucket_name)

        train_valObj.train_validation()

        trainModelObj = Train_Model()

        num_clusters = trainModelObj.training_model()

        loadProdModelObj = Load_Prod_Model(num_clusters)

        loadProdModelObj.load_production_model()

        return Response("Training successfull!!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.get("/predict")
async def predictRouteClient():
    try:
        raw_data_pred_bucket_name = config["s3_bucket"]["phising_raw_data_bucket"]

        pred_val = Pred_Validation(raw_data_pred_bucket_name)

        pred_val.prediction_validation()

        pred = Prediction()

        bucket, filename, json_predictions = pred.predict_from_model()

        return Response(
            f"Prediction file created in {bucket} bucket with filename as {filename}, and few of the predictions are {str(json.loads(json_predictions))}"
        )

    except Exception as e:
        return Response(f"Error Occurred! {e}")


if __name__ == "__main__":
    host = config["app"]["host"]

    port = config["app"]["port"]

    uvicorn.run(app, host=host, port=port)
