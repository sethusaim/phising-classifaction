from phising.data_ingestion.data_loader_train import Data_Getter_Train
from phising.data_preprocessing.clustering import KMeans_Clustering
from phising.data_preprocessing.preprocessing import Preprocessor
from phising.mlflow_utils.mlflow_operations import MLFlow_Operation
from phising.s3_bucket_operations.s3_operations import S3_Operation
from utils.logger import App_Logger
from utils.model_utils import Model_Utils
from utils.read_params import read_params


class Train_Model:
    """
    Description :   This method is used for getting the data and applying
                    some preprocessing steps and then train the models and register them in mlflow
    Written by  :   iNeuron Intelligence
    
    Version     :   1.2
    Revisions   :   Moved to setup to cloud 
    """

    def __init__(self):
        self.log_writer = App_Logger()

        self.config = read_params()

        self.model_train_log = self.config["train_db_log"]["train_model"]

        self.target_col = self.config["target_col"]

        self.class_name = self.__class__.__name__

        self.mlflow_op = MLFlow_Operation(self.model_train_log)

        self.data_getter_train = Data_Getter_Train(self.model_train_log)

        self.preprocessor = Preprocessor(self.model_train_log)

        self.kmeans_op = KMeans_Clustering(self.model_train_log)

        self.model_utils = Model_Utils()

        self.s3 = S3_Operation()

    def training_model(self):
        """
        Method Name :   training_model
        Description :   This method is responsible for applying the preprocessing functions and then train models againist 
                        training data and them register them in mlflow

        Output      :   A pandas series object consisting of runs for the particular experiment id
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Written by  :   iNeuron Intelligence
        Revisions   :   moved setup to cloud
        """
        method_name = self.training_model.__name__

        self.log_writer.start_log(
            "start", self.class_name, method_name, self.model_train_log,
        )

        try:
            data = self.data_getter_train.get_data()

            data = self.preprocessor.replace_invalid_values(data)

            is_null_present = self.preprocessor.is_null_present(X)

            if is_null_present:
                data = self.preprocessor.impute_missing_values(X)

            X, Y = self.preprocessor.separate_label_feature(
                data, label_column_name=self.target_col
            )

            number_of_clusters = self.kmeans_op.elbow_plot(X)

            X, kmeans_model = self.kmeans_op.create_clusters(
                data=X, number_of_clusters=number_of_clusters
            )

            X["Labels"] = Y

            list_of_clusters = X["Cluster"].unique()

            for i in list_of_clusters:
                cluster_data = X[X["Cluster"] == i]

                cluster_features = cluster_data.drop(["Labels", "Cluster"], axis=1)

                cluster_label = cluster_data["Labels"]

                self.log_writer.log(
                    "Seprated cluster features and cluster label for the cluster data",
                    self.model_train_log,
                )

                self.model_utils.train_and_log_models(
                    cluster_features,
                    cluster_label,
                    self.model_train_log,
                    idx=i,
                    kmeans=kmeans_model,
                )

            self.log_writer.log("Successful End of Training", self.model_train_log)

            self.log_writer.start_log(
                "exit", self.class_name, method_name, self.model_train_log,
            )

            return number_of_clusters

        except Exception as e:
            self.log_writer.log("Unsuccessful End of Training", self.model_train_log)

            self.log_writer.exception_log(
                e, self.class_name, method_name, self.model_train_log,
            )
