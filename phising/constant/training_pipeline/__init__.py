import os
from datetime import datetime

import numpy as np

TIMESTAMP: datetime = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

PIPELINE_NAME: str = "network"

EXP_NAME: str = f"{PIPELINE_NAME}-{TIMESTAMP}"

TARGET_COLUMN: str = "Result"

ARTIFACT_DIR: str = "artifacts"

LOG_DIR: str = "logs"

TRAIN_FILE_NAME: str = "train.csv"

TEST_FILE_NAME: str = "test.csv"

MODEL_FILE_NAME: str = "model.pkl"

APP_ARTIFACTS_BUCKET: str = "12272network-artifacts"

SCHEMA_FILE_PATH: str = os.path.join("config", "network_schema_prediction.yaml")

PREPROCSSING_OBJECT_FILE_NAME: str = "preprocessing.pkl"

"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_DIR_NAME: str = "data_ingestion"

DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"

DATA_INGESTION_INGESTED_DIR: str = "ingested"

DATA_INGESTION_BUCKET_NAME: str = "36930network-feature-store"

DATA_INGESTION_BUCKET_FOLDER_NAME: str = "data/train_batch"

"""
Data Validation realted contant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"

DATA_VALIDATION_VALID_DIR: str = "valid"

DATA_VALIDATION_INVALID_DIR: str = "invalid"

DATA_VALIDATION_TEST_SIZE: float = 0.3

DATA_VALIDATION_TRAIN_SCHEMA: str = "config/network_schema_training.yaml"

DATA_VALIDATION_REGEX: str = "config/network_regex.txt"

DATA_VALIDATION_TRAIN_COMPRESSED_FILE_PATH: str = "train_input_file.csv"

DATA_VALIDATION_TRAIN_FILE_PATH: str = "train.csv"

DATA_VALIDATION_TEST_FILE_PATH: str = "test.csv"

"""
Data Transformation ralated constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"

DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"

DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
    "missing_values": np.nan,
    "n_neighbors": 3,
    "weights": "uniform",
}

DATA_TRANSFORMATION_TRAIN_FILE_PATH: str = "train.npy"

DATA_TRANSFORMATION_TEST_FILE_PATH: str = "test.npy"

"""
Model Trainer ralated constant start with MODE TRAINER VAR NAME
"""

MODEL_TRAINER_DIR_NAME: str = "model_trainer"

MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"

MODEL_TRAINER_BEST_MODEL_DIR: str = "best_model"

MODEL_TRAINER_EXPECTED_SCORE: float = 0.6

MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD: float = 0.02

MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = "config/model.yaml"

MODEL_TRAINER_MODEL_METRIC_KEY: str = "roc_auc_score"

"""
MODEL Evauation related constant start with MODEL_EVALUATION var name
"""
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.03

MODEL_EVALUATION_THRESHOLD: float = 0.6

MODEL_EVALUATION_MODEL_TYPE: str = "classifier"

"""
MODEL Pusher related constant start with MODEL_PUSHER var name
"""
MODEL_PUSHER_PROD_MODEL_STAGE: str = "Production"

MODEL_PUSHER_STAG_MODEL_STAGE: str = "Staging"

MODEL_PUSHER_ARCHIVE_EXISTING_VERSIONS: bool = True

MODEL_PUSHER_BENTOML_MODEL_NAME: str = "network-model"

MODEL_PUSHER_BENTOML_SERVICE_NAME: str = "network_model_service"

MODEL_PUSHER_BENTOML_MODEL_IMAGE: str = "networkimage"

MODEL_PUSHER_MODEL_ECR_URI: str = ""
