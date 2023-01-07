import sys

from phising.exception import PhisingException
from phising.pipeline.training_pipeline import TrainPipeline
from phising.utils.main_utils import sync_app_artifacts


def start_training():
    try:
        tp = TrainPipeline()

        tp.run_pipeline()

    except Exception as e:
        raise PhisingException(e, sys)

    finally:
        sync_app_artifacts()


if __name__ == "__main__":
    start_training()
