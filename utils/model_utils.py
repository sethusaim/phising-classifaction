from mlflow import start_run
from phising.mlflow_utils.mlflow_operations import MLFlow_Operation
from phising.model_finder.tuner import Model_Finder
from phising.s3_bucket_operations.s3_operations import S3_Operation
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split

from utils.logger import App_Logger
from utils.read_params import read_params


class Model_Utils:
    """
    Description :   This class is used for all the model utils
    
    Version     :   1.2
    Revisions   :   Moved to setup to cloud 
    """

    def __init__(self):
        self.log_writer = App_Logger()

        self.config = read_params()

        self.tuner_kwargs = self.config["model_utils"]

        self.split_kwargs = self.config["base"]

        self.train_model_dir = self.config["models_dir"]["trained"]

        self.save_format = self.config["model_save_format"]

        self.model_bucket = self.config["s3_bucket"]["model"]

        self.exp_name = self.config["mlflow_config"]["experiment_name"]

        self.run_name = self.config["mlflow_config"]["run_name"]

        self.mlflow_op = MLFlow_Operation()

        self.model_finder = Model_Finder()

        self.s3 = S3_Operation()

        self.class_name = self.__class__.__name__

    def get_model_score(self, model, test_x, test_y, log_file):
        """
        Method Name :   get_model_score
        Description :   This method gets model score againist the test data

        Output      :   A model score is returned 
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """

        method_name = self.get_model_score.__name__

        self.log_writer.start_log("start", self.class_name, method_name, log_file)

        try:
            model_name = model.__class__.__name__

            preds = model.predict(test_x)

            self.log_writer.log(
                log_file, f"Used {model_name} model to get predictions on test data"
            )

            if len(test_y.unique()) == 1:
                model_score = accuracy_score(test_y, preds)

                self.log_writer.log(
                    log_file, f"Accuracy for {model_name} is {model_score}"
                )

            else:
                model_score = roc_auc_score(test_y, preds)

                self.log_writer.log(
                    log_file, f"AUC score for {model_name} is {model_score}"
                )

            self.log_writer.start_log("exit", self.class_name, method_name, log_file)

            return model_score

        except Exception as e:
            self.log_writer.exception_log(e, self.class_name, method_name, log_file)

    def get_model_params(self, model, x_train, y_train, log_file):
        """
        Method Name :   get_model_params
        Description :   This method gets the model parameters based on model_key_name and train data

        Output      :   Best model parameters are returned
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """

        method_name = self.get_model_params.__name__

        self.log_writer.start_log("start", self.class_name, method_name, log_file)

        try:
            model_name = model.__class__.__name__

            model_param_grid = self.config[model_name]

            model_grid = GridSearchCV(
                estimator=model, param_grid=model_param_grid, **self.tuner_kwargs
            )

            self.log_writer.log(
                log_file,
                f"Initialized {model_grid.__class__.__name__}  with {model_param_grid} as params",
            )

            model_grid.fit(x_train, y_train)

            self.log_writer.log(
                log_file,
                f"Found the best params for {model_name} model based on {model_param_grid} as params",
            )

            self.log_writer.start_log("exit", self.class_name, method_name, log_file)

            return model_grid.best_params_

        except Exception as e:
            self.log_writer.exception_log(e, self.class_name, method_name, log_file)

    def train_and_log_models(self, X_data, Y_data, log_file, idx=None, kmeans=None):
        method_name = self.train_and_log_models.__name__

        self.log_writer.start_log("start", log_file, self.class_name, method_name)

        try:
            x_train, x_test, y_train, y_test = train_test_split(
                X_data, Y_data, **self.split_kwargs
            )

            self.log_writer.log(
                log_file,
                f"Performed train test split with kwargs as {self.split_kwargs}",
            )

            lst = self.model_finder.get_trained_models(x_train, y_train, x_test, y_test)

            self.log_writer.log(log_file, "Got trained models")

            for _, tm in enumerate(lst):
                self.s3.save_model(
                    tm[0],
                    self.train_model_dir,
                    self.model_bucket,
                    log_file,
                    format=self.save_format,
                )

                self.mlflow_op.set_mlflow_tracking_uri()

                self.mlflow_op.set_mlflow_experiment(self.exp_name)

                with start_run(run_name=self.run_name):
                    self.mlflow_op.log_all_for_model(idx, tm[0], tm[1])

                    if kmeans is not None:
                        self.mlflow_op.log_all_for_model(None, kmeans, None)

                    else:
                        pass

            self.log_writer.log(
                log_file, "Saved and logged all trained models to mlflow"
            )

            self.log_writer.start_log("exit", log_file, self.class_name, method_name)

        except Exception as e:
            self.log_writer.exception_log(e, log_file, self.class_name, method_name)
