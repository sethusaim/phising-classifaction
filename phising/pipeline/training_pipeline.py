import sys
import os

from phising.components.data_ingestion import DataIngestion
from phising.components.data_transformation import DataTransformation
from phising.components.data_validation import DataValidation
from phising.components.model_evaluation import ModelEvaluation
from phising.components.model_pusher import ModelPusher
from phising.components.model_trainer import ModelTrainer
from phising.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact,
    ModelTrainerArtifact,
)
from phising.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelEvaluationConfig,
    ModelPusherConfig,
    ModelTrainerConfig,
    TrainingPipelineConfig,
)
from phising.exception import NetworkException


class TrainPipeline:
    is_pipeline_running = False

    def __init__(self):
        self.training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            self.data_ingestion_config: DataIngestionConfig = DataIngestionConfig(
                training_pipeline_config=self.training_pipeline_config
            )

            data_ingestion: DataIngestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )

            data_ingestion_artifact: DataIngestionArtifact = (
                data_ingestion.initiate_data_ingestion()
            )

            return data_ingestion_artifact

        except Exception as e:
            raise NetworkException(e, sys)

    def start_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        try:
            self.data_validation_config: DataValidationConfig = DataValidationConfig(
                training_pipeline_config=self.training_pipeline_config
            )

            data_validation: DataValidation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config,
            )

            data_validation_artifact: DataValidationArtifact = (
                data_validation.initiate_data_validation()
            )

            return data_validation_artifact

        except Exception as e:
            raise NetworkException(e, sys)

    def start_data_transformation(
        self, data_validation_artifact: DataValidationArtifact
    ) -> DataTransformationArtifact:
        try:
            self.data_transformation_config: DataTransformationConfig = (
                DataTransformationConfig(
                    training_pipeline_config=self.training_pipeline_config
                )
            )

            data_transformation: DataTransformation = DataTransformation(
                data_validation_artifact=data_validation_artifact,
                data_transformation_config=self.data_transformation_config,
            )

            data_transformation_artifact: DataTransformationArtifact = (
                data_transformation.initiate_data_transformation()
            )

            return data_transformation_artifact

        except Exception as e:
            raise NetworkException(e, sys)

    def start_model_trainer(
        self, data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        try:
            self.model_trainer_config: ModelTrainerConfig = ModelTrainerConfig(
                training_pipeline_config=self.training_pipeline_config
            )

            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config,
            )

            model_trainer_artifact = model_trainer.initiate_model_trainer()

            return model_trainer_artifact

        except Exception as e:
            raise NetworkException(e, sys)

    def start_model_evaluation(
        self,
        data_validation_artifact: DataValidationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ) -> ModelEvaluationArtifact:
        try:
            self.model_eval_config: ModelEvaluationConfig = ModelEvaluationConfig()

            model_evaluation = ModelEvaluation(
                model_eval_config=self.model_eval_config,
                data_validation_artifact=data_validation_artifact,
                model_trainer_artifact=model_trainer_artifact,
            )

            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()

            return model_evaluation_artifact

        except Exception as e:
            raise NetworkException(e, sys)

    def start_model_pusher(self, model_evaluation_artifact: ModelEvaluationArtifact):
        try:
            self.model_pusher_config: ModelPusherConfig = ModelPusherConfig()

            model_pusher = ModelPusher(
                model_evaluation_artifact=model_evaluation_artifact,
                model_pusher_config=self.model_pusher_config,
            )

            model_pusher_artifact = model_pusher.initiate_model_pusher()

            return model_pusher_artifact

        except Exception as e:
            raise NetworkException(e, sys)

    def run_pipeline(self):
        try:
            TrainPipeline.is_pipeline_running = True

            data_ingestion_artifact: DataIngestionArtifact = self.start_data_ingestion()

            data_validation_artifact: DataValidationArtifact = (
                self.start_data_validation(
                    data_ingestion_artifact=data_ingestion_artifact
                )
            )

            data_transformation_artifact: DataTransformationArtifact = (
                self.start_data_transformation(
                    data_validation_artifact=data_validation_artifact
                )
            )

            model_trainer_artifact: ModelTrainerArtifact = self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact
            )

            model_evaluation_artifact: ModelEvaluationArtifact = (
                self.start_model_evaluation(
                    data_validation_artifact=data_validation_artifact,
                    model_trainer_artifact=model_trainer_artifact,
                )
            )

            model_pusher_artifact: ModelPusherArtifact = self.start_model_pusher(
                model_evaluation_artifact=model_evaluation_artifact
            )

        except Exception as e:
            raise NetworkException(e, sys)
