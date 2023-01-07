import os
import sys
from typing import Dict, Union

import bentoml
import dill
import numpy as np
import yaml

from phising.cloud_storage.aws_operations import S3Sync
from phising.constant import training_pipeline
from phising.exception import PhisingException
from phising.logger import logging


def read_yaml(file_name: str) -> Dict:
    logging.info("Entered the read_yaml class of MainUtils class")

    try:
        with open(file_name) as f:
            dic: Dict = yaml.safe_load(f)

        logging.info(f"Read the yaml content from {file_name}")

        logging.info("Exited the read_yaml class of MainUtils class")

        return dic

    except Exception as e:
        raise PhisingException(e, sys)


def read_text(file_name: str) -> str:
    logging.info("Entered the read_text class of MainUtils class")

    try:
        with open(file_name, "r") as f:
            txt: str = f.read()

        logging.info(f"Read the text content from {file_name}")

        logging.info("Exited the read_text class of MainUtils class")

        return txt

    except Exception as e:
        raise PhisingException(e, sys)


def save_numpy_array_data(file_path: str, array: Union[np.array, np.ndarray]) -> None:
    logging.info("Entered the save_numpy_array_data class of MainUtils class")

    try:
        dir_path: str = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)

        logging.info(f"Saved {array} numpy array to {file_path}")

        logging.info("Exited the save_numpy_array_data class of MainUtils class")

    except Exception as e:
        raise PhisingException(e, sys)


def load_numpy_array_data(file_path: str) -> Union[np.array, np.ndarray]:
    logging.info("Entered the load_numpy_array_data class of MainUtils class")

    try:
        with open(file_path, "rb") as file_obj:
            obj = np.load(file_obj, allow_pickle=True)

        logging.info(f"Loaded numpy array from {file_path}")

        logging.info("Exited the load_numpy_array_data class of MainUtils class")

        return obj

    except Exception as e:
        raise PhisingException(e, sys)


def load_object(file_path: str) -> object:
    logging.info("Entered the load_object method of MainUtils class")

    try:
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)

        logging.info(f"Loaded object from {file_path}")

        logging.info("Exited the load_object method of MainUtils class")

        return obj

    except Exception as e:
        raise PhisingException(e, sys)


def save_object(file_path: str, obj: object) -> None:
    logging.info("Entered the save_object method of MainUtils class")

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info("Exited the save_object method of MainUtils class")

    except Exception as e:
        raise PhisingException(e, sys)


def sync_app_artifacts() -> None:
    try:
        s3 = S3Sync()

        s3.sync_folder_to_s3(
            folder=training_pipeline.ARTIFACT_DIR,
            bucket_name=training_pipeline.APP_ARTIFACTS_BUCKET,
            bucket_folder_name=training_pipeline.PIPELINE_NAME
            + "/"
            + training_pipeline.ARTIFACT_DIR,
        )

        s3.sync_folder_to_s3(
            folder=training_pipeline.LOG_DIR,
            bucket_name=training_pipeline.APP_ARTIFACTS_BUCKET,
            bucket_folder_name=training_pipeline.PIPELINE_NAME
            + "/"
            + training_pipeline.LOG_DIR,
        )

    except Exception as e:
        raise PhisingException(e, sys)


def build_and_push_bento_image(model_uri: str) -> None:
    try:
        bentoml.mlflow.import_model(
            name=training_pipeline.MODEL_PUSHER_BENTOML_MODEL_NAME, model_uri=model_uri
        )

        os.system(
            f"bentoml containerize {training_pipeline.MODEL_PUSHER_BENTOML_SERVICE_NAME}:latest {training_pipeline.MODEL_PUSHER_MODEL_ECR_URI}"
        )

    except Exception as e:
        raise PhisingException(e, sys)
