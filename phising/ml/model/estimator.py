import sys

from mlflow.pyfunc import PythonModel
from pandas import DataFrame
from sklearn.pipeline import Pipeline

from phising.exception import NetworkException


class NetworkModel(PythonModel):
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        self.preprocessing_object = preprocessing_object

        self.trained_model_object = trained_model_object

    def predict(self, context, dataframe: DataFrame) -> DataFrame:
        try:
            transformed_feature = self.preprocessing_object.transform(dataframe)

            preds = self.trained_model_object.predict(transformed_feature)

            return preds

        except Exception as e:
            raise NetworkException(e, sys)

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"
