import sys

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score

from phising.entity.artifact_entity import ClassificationMetricArtifact
from phising.exception import NetworkException


def calculate_metric(
    model: BaseEstimator, x: pd.DataFrame, y: pd.DataFrame
) -> ClassificationMetricArtifact:
    try:
        yhat = model.predict(x)

        classification_metric: float = roc_auc_score(y, yhat)

        return classification_metric

    except Exception as e:
        raise NetworkException(e, sys)
