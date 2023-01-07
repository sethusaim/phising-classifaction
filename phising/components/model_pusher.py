import sys

from phising.configuration.mlflow_connection import MLFlowClient
from phising.entity.artifact_entity import ModelEvaluationArtifact, ModelPusherArtifact
from phising.entity.config_entity import ModelPusherConfig
from phising.exception import PhisingException
from phising.logger import logging
from phising.utils.main_utils import build_and_push_bento_image


class ModelPusher:
    def __init__(
        self,
        model_evaluation_artifact: ModelEvaluationArtifact,
        model_pusher_config: ModelPusherConfig,
    ):
        self.model_evaluation_artifact = model_evaluation_artifact

        self.model_pusher_config = model_pusher_config

        self.mlflow_client = MLFlowClient().client

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        logging.info("Entered initiate_model_pusher method of ModelPusher class")

        try:
            if self.model_evaluation_artifact.accepted_model_info is None:
                raise Exception("No trained model is accepted")

            elif (
                self.model_evaluation_artifact.accepted_model_info is not None
                and self.model_evaluation_artifact.prod_model_info is None
            ):
                logging.info(
                    "Production model info is None, Accepted model info is not None. Moving accepted model to production"
                )

                self.mlflow_client.transition_model_version_stage(
                    name=self.model_evaluation_artifact.accepted_model_info.model_name,
                    version=self.model_evaluation_artifact.accepted_model_info.model_version,
                    stage=self.model_pusher_config.production_model_stage,
                    archive_existing_versions=self.model_pusher_config.archive_existing_versions,
                )

                build_and_push_bento_image(
                    model_uri=self.model_evaluation_artifact.accepted_model_info.model_uri
                )

            elif (
                self.model_evaluation_artifact.accepted_model_info is not None
                and self.model_evaluation_artifact.prod_model_info is not None
            ):
                logging.info(
                    "Accepted model info is not None and Production model info is not None. Moving accepted model to production and production model to staging"
                )

                self.mlflow_client.transition_model_version_stage(
                    name=self.model_evaluation_artifact.accepted_model_info.model_name,
                    version=self.model_evaluation_artifact.accepted_model_info.model_version,
                    stage=self.model_pusher_config.production_model_stage,
                    archive_existing_versions=self.model_pusher_config.archive_existing_versions,
                )

                self.mlflow_client.transition_model_version_stage(
                    name=self.model_evaluation_artifact.prod_model_info.model_name,
                    version=self.model_evaluation_artifact.prod_model_info.model_version,
                    stage=self.model_pusher_config.staging_model_stage,
                    archive_existing_versions=self.model_pusher_config.archive_existing_versions,
                )

                build_and_push_bento_image(
                    model_uri=self.model_evaluation_artifact.accepted_model_info.model_uri
                )

            else:
                logging.info("something went wrong")

            logging.info("Exited initiate_model_pusher method of ModelPusher class")

        except Exception as e:
            raise PhisingException(e, sys)
