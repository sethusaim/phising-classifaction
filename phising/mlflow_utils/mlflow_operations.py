from os import environ

from mlflow import (
    get_experiment_by_name,
    log_metric,
    log_param,
    search_runs,
    set_experiment,
    set_tracking_uri,
)
from mlflow.sklearn import log_model
from mlflow.tracking import MlflowClient
from phising.s3_bucket_operations.s3_operations import S3_Operation
from utils.logger import App_Logger
from utils.read_params import read_params


class MLFlow_Operation:
    """
    Description :    This class shall be used for handling all the mlflow operations

    Version     :   1.2
    Revisions   :   Moved to setup to cloud 
    """

    def __init__(self, log_file):
        self.config = read_params()

        self.class_name = self.__class__.__name__

        self.log_writer = App_Logger()

        self.s3 = S3_Operation()

        self.log_file = log_file

        self.mlflow_save_format = self.config["mlflow_config"]["serialization_format"]

        self.trained_models_dir = self.config["models_dir"]["trained"]

        self.staged_models_dir = self.config["models_dir"]["stag"]

        self.prod_models_dir = self.config["models_dir"]["prod"]

        self.model_save_format = self.config["model_utils"]["save_format"]

        self.remote_server_uri = environ["MLFLOW_TRACKING_URI"]

    def get_experiment_from_mlflow(self, exp_name):
        """
        Method Name :   get_experiment_from_mlflow
        Description :   This method gets the experiment from mlflow server using the experiment name

        Output      :   An experiment which was stored in mlflow server
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_experiment_from_mlflow.__name__

        self.log_writer.start_log("start", self.class_name, method_name, self.log_file)

        try:
            exp = get_experiment_by_name(exp_name)

            self.log_writer.log(f"Got {exp_name} experiment from mlflow", self.log_file)

            self.log_writer.start_log(
                "exit", self.class_name, method_name, self.log_file
            )

            return exp

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name, self.log_file
            )

    def get_runs_from_mlflow(self, exp_id):
        """
        Method Name :   get_runs_from_mlflow
        Description :   This method gets the runs from the mlflow server for a particular experiment id

        Output      :   A pandas series object consisting of runs for the particular experiment id
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_runs_from_mlflow.__name__

        self.log_writer.start_log("start", self.class_name, method_name, self.log_file)

        try:
            runs = search_runs(experiment_ids=exp_id)

            self.log_writer.log(
                f"Completed searching for runs in mlflow with experiment ids as {exp_id}",
                self.log_file,
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name, self.log_file
            )

            return runs

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name, self.log_file
            )

    def set_mlflow_experiment(self, experiment_name):
        """
        Method Name :   set_mlflow_experiment
        Description :   This method sets the mlflow experiment with the particular experiment name

        Output      :   An experiment with experiment name will be created in mlflow server
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        
        Revisions   :   moved setup to cloud
        """
        method_name = self.set_mlflow_experiment.__name__

        self.log_writer.start_log("start", self.class_name, method_name, self.log_file)

        try:
            set_experiment(experiment_name)

            self.log_writer.log(
                f"Set mlflow experiment with name as {experiment_name}", self.log_file
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name, self.log_file
            )

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name, self.log_file
            )

    def get_mlflow_client(self):
        """
        Method Name :   get_mlflow_client
        Description :   This method gets mlflow client for the particular server uri

        Output      :   A mlflow client is created with particular server uri
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_mlflow_client.__name__

        self.log_writer.start_log("start", self.class_name, method_name, self.log_file)

        try:
            client = MlflowClient(self.remote_server_uri)

            self.log_writer.log("Got mlflow client with tracking uri", self.log_file)

            self.log_writer.start_log(
                "exit", self.class_name, method_name, self.log_file
            )

            return client

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name, self.log_file
            )

    def set_mlflow_tracking_uri(self):
        """
        Method Name :   set_mlflow_tracking_uri
        Description :   This method sets the mlflow tracking uri in mlflow server 

        Output      :   MLFLow server will set the particular uri to communicate with code 
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        
        Revisions   :   moved setup to cloud
        """
        method_name = self.set_mlflow_tracking_uri.__name__

        self.log_writer.start_log("start", self.class_name, method_name, self.log_file)

        try:
            set_tracking_uri(self.remote_server_uri)

            self.log_writer.log(self.log_file, "Set mlflow tracking uri")

            self.log_writer.start_log(
                "exit", self.class_name, method_name, self.log_file
            )

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name, self.log_file
            )

    def get_mlflow_models(self):
        """
        Method Name :   get_mlflow_models
        Description :   This method gets the registered models in mlflow server

        Output      :   A list of registered model names stored in mlflow server
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_mlflow_models.__name__

        self.log_writer.start_log("start", self.class_name, method_name, self.log_file)

        try:
            client = self.get_mlflow_client()

            reg_model_names = [rm.name for rm in client.list_registered_models()]

            self.log_writer.log("Got registered models from mlflow", self.log_file)

            self.log_writer.start_log(
                "exit", self.class_name, method_name, self.log_file
            )

            return reg_model_names

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name, self.log_file
            )

    def search_mlflow_models(self, order):
        """
        Method Name :   search_mlflow_models
        Description :   This method searches for registered models and returns them in the mentioned order

        Output      :   A list of registered models in the mentioned order
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        
        Revisions   :   moved setup to cloud
        """
        method_name = self.search_mlflow_models.__name__

        self.log_writer.start_log("start", self.class_name, method_name, self.log_file)

        try:
            client = self.get_mlflow_client()

            results = client.search_registered_models(order_by=[f"name {order}"])

            self.log_writer.log(
                f"Got registered models in mlflow in {order} order", self.log_file
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name, self.log_file
            )

            return results

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name, self.log_file
            )

    def log_sklearn_model(self, model, model_name):
        """
        Method Name :   log_sklearn_model
        Description :   This method logs the model to mlflow server

        Output      :   A model is logged to the mlflow server
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        
        Revisions   :   moved setup to cloud
        """
        method_name = self.log_sklearn_model.__name__

        self.log_writer.start_log("start", self.class_name, method_name, self.log_file)

        try:
            log_model(
                sk_model=model,
                serialization_format=self.mlflow_save_format,
                registered_model_name=model_name,
                artifact_path=model_name,
            )

            self.log_writer.log(f"Logged {model_name} model in mlflow", self.log_file)

            self.log_writer.start_log(
                "exit", self.class_name, method_name, self.log_file
            )

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name, self.log_file
            )

    def log_model_metric(self, model_name, metric):
        """
        Method Name :   log_model_metric
        Description :   This method logs the model metric to mlflow server

        Output      :   A model metric is logged to mlflow server
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        
        Revisions   :   moved setup to cloud
        """
        method_name = self.log_model_metric.__name__

        self.log_writer.start_log("start", self.class_name, method_name, self.log_file)

        try:
            model_score_name = f"{model_name}-best_score"

            log_metric(model_score_name, metric)

            self.log_writer.log(
                self.log_file, f"{model_score_name} logged in mlflow",
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name, self.log_file
            )

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name, self.log_file
            )

    def log_model_param(self, idx, model, model_name, param):
        """
        Method Name :   log_model_param
        Description :   This method logs the model param to mlflow server

        Output      :   A model param is logged to mlflow server
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        
        Revisions   :   moved setup to cloud
        """
        method_name = self.log_model_param.__name__

        self.log_writer.start_log("start", self.class_name, method_name, self.log_file)

        try:
            model_param_name = model_name + str(idx) + f"-{param}"

            log_param(model_param_name, model.__dict__[param])

            self.log_writer.log(f"{model_param_name} logged in mlflow", self.log_file)

            self.log_writer.start_log(
                "exit", self.class_name, method_name, self.log_file
            )

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name, self.log_file
            )

    def log_all_for_model(self, model, model_score, idx=None):
        """
        Method Name :   log_all_for_model
        Description :   This method logs model,model params and model score to mlflow server

        Output      :   Model,model parameters and model score are logged to mlflow server
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        
        Revisions   :   moved setup to cloud
        """
        method_name = self.log_all_for_model.__name__

        try:
            self.log_writer.start_log(
                "start", self.class_name, method_name, self.log_file,
            )

            base_model_name = model.__class__.__name__

            if base_model_name is "KMeans":
                self.log_sklearn_model(model, base_model_name)

            else:
                model_name = base_model_name + str(idx)

                self.log_writer.log(
                    self.log_file, f"Got the model name as {model_name}",
                )

                model_params_list = list(self.config[model_name].keys())

                self.log_writer.log(
                    self.log_file, f"Created a list of params based on {model_name}",
                )

                for param in model_params_list:
                    self.log_model_param(idx, model, model_name, param=param)

                self.log_sklearn_model(model, model_name)

                self.log_model_metric(model_name, metric=float(model_score))

            self.log_writer.start_log(
                "exit", self.class_name, method_name, self.log_file
            )

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name, self.log_file
            )

    def transition_mlflow_model(
        self, model_version, stage, model_name, from_bucket, to_bucket
    ):
        """
        Method Name :   transition_mlflow_model
        Description :   This method transitions mlflow model from one stage to other stage, and does the same in s3 bucket

        Output      :   A mlflow model is transitioned from one stage to another, and same is reflected in s3 bucket
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        
        Revisions   :   moved setup to cloud
        """
        method_name = self.transition_mlflow_model.__name__

        self.log_writer.start_log("start", self.class_name, method_name, self.log_file)

        try:
            current_version = model_version

            self.log_writer.log(
                self.log_file, f"Got {current_version} as the current model version",
            )

            client = self.get_mlflow_client()

            trained_model_file = (
                self.trained_models_dir + "/" + model_name + self.model_save_format
            )

            stag_model_file = (
                self.staged_models_dir + "/" + model_name + self.model_save_format
            )

            prod_model_file = (
                self.prod_models_dir + "/" + model_name + self.model_save_format
            )

            self.log_writer.log(
                self.log_file, "Created trained,stag and prod model files",
            )

            if stage == "Production":
                self.log_writer.log(
                    self.log_file, f"{stage} is selected for transition",
                )

                client.transition_model_version_stage(
                    name=model_name, version=current_version, stage=stage
                )

                self.log_writer.log(
                    self.log_file, f"Transitioned {model_name} to {stage} in mlflow",
                )

                self.s3.copy_data(
                    trained_model_file,
                    from_bucket,
                    prod_model_file,
                    to_bucket,
                    self.log_file,
                )

            elif stage == "Staging":
                self.log_writer.log(
                    self.log_file, f"{stage} is selected for transition",
                )

                client.transition_model_version_stage(
                    name=model_name, version=current_version, stage=stage
                )

                self.log_writer.log(
                    self.log_file, f"Transitioned {model_name} to {stage} in mlflow",
                )

                self.s3.copy_data(
                    trained_model_file,
                    from_bucket,
                    stag_model_file,
                    to_bucket,
                    self.log_file,
                )

            else:
                self.log_writer.log(
                    self.log_file, "Please select stage for model transition",
                )

            self.log_writer.start_log(
                "exit", self.class_name, method_name, self.log_file
            )

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name, self.log_file
            )
