import mlflow
from phising.data_ingestion.data_loader_train import data_getter_train
from phising.data_preprocessing.clustering import kmeans_clustering
from phising.data_preprocessing.preprocessing import preprocessor
from phising.mlflow_utils.mlflow_operations import mlflow_operations
from phising.model_finder.tuner import model_finder
from phising.s3_bucket_operations.s3_operations import s3_operations
from sklearn.model_selection import train_test_split
from utils.logger import App_Logger
from utils.read_params import read_params


class train_model:
    """
    Description :   This method is used for getting the data and applying
                    some preprocessing steps and then train the models and register them in mlflow

    Version     :   1.2
    Revisions   :   moved to setup to cloud
    """

    def __init__(self):
        self.log_writer = App_Logger()

        self.config = read_params()

        self.model_train_log = self.config["train_db_log"]["model_training"]

        self.model_bucket = self.config["s3_bucket"]["phising_model_bucket"]

        self.test_size = self.config["base"]["test_size"]

        self.target_col = self.config["base"]["target_col"]

        self.random_state = self.config["base"]["random_state"]

        self.remote_server_uri = self.config["mlflow_config"]["remote_server_uri"]

        self.experiment_name = self.config["mlflow_config"]["experiment_name"]

        self.run_name = self.config["mlflow_config"]["run_name"]

        self.train_model_dir = self.config["models_dir"]["trained"]

        self.class_name = self.__class__.__name__

        self.mlflow_op = mlflow_operations(table_name=self.model_train_log)

        self.data_getter_train_obj = data_getter_train(table_name=self.model_train_log)

        self.preprocessor_obj = preprocessor(table_name=self.model_train_log)

        self.kmeans_obj = kmeans_clustering(table_name=self.model_train_log)

        self.model_finder_obj = model_finder(table_name=self.model_train_log)

        self.s3 = s3_operations()

    def training_model(self):
        """
        Method Name :   training_model
        Description :   This method is used for getting the data and applying
                        some preprocessing steps and then train the models and register them in mlflow

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.training_model.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=self.model_train_log,
        )

        try:
            data = self.data_getter_train_obj.get_data()

            data = self.preprocessor_obj.remove_columns(data, ["phising"])

            X, Y = self.preprocessor_obj.separate_label_feature(
                data, label_column_name=self.target_col
            )

            is_null_present = self.preprocessor_obj.is_null_present(X)

            if is_null_present:
                X = self.preprocessor_obj.impute_missing_values(X)

            cols_to_drop = self.preprocessor_obj.get_columns_with_zero_std_deviation(X)

            X = self.preprocessor_obj.remove_columns(X, cols_to_drop)

            number_of_clusters = self.kmeans_obj.elbow_plot(X)

            X, kmeans_model = self.kmeans_obj.create_clusters(
                data=X, number_of_clusters=number_of_clusters
            )

            X["Labels"] = Y

            list_of_clusters = X["Cluster"].unique()

            for i in list_of_clusters:
                cluster_data = X[X["Cluster"] == i]

                cluster_features = cluster_data.drop(["Labels", "Cluster"], axis=1)

                cluster_label = cluster_data["Labels"]

                self.log_writer.log(
                    table_name=self.model_train_log,
                    log_message="Seprated cluster features and cluster label for the cluster data",
                )

                x_train, x_test, y_train, y_test = train_test_split(
                    cluster_features,
                    cluster_label,
                    test_size=self.test_size,
                    random_state=self.random_state,
                )

                self.log_writer.log(
                    table_name=self.model_train_log,
                    log_message=f"Performed train test split with test size as {self.test_size} and random state as {self.random_state}",
                )

                (
                    xgb_model,
                    xgb_model_score,
                    rf_model,
                    rf_model_score,
                ) = self.model_finder_obj.get_trained_models(
                    x_train, y_train, x_test, y_test
                )

                self.s3.save_model(
                    idx=i,
                    model=xgb_model,
                    model_bucket=self.model_bucket,
                    table_name=self.model_train_log,
                    model_dir="",
                )

                self.s3.save_model(
                    idx=i,
                    model=rf_model,
                    model_bucket=self.model_bucket,
                    table_name=self.model_train_log,
                    model_dir="",
                )

                try:
                    self.mlflow_op.set_mlflow_tracking_uri(
                        server_uri=self.remote_server_uri
                    )

                    self.mlflow_op.set_mlflow_experiment(
                        experiment_name=self.experiment_name
                    )

                    with mlflow.start_run(run_name=self.run_name):
                        self.mlflow_op.log_all_for_model(
                            idx=None,
                            model=kmeans_model,
                            model_param_name=None,
                            model_score=None,
                        )

                        self.mlflow_op.log_all_for_model(
                            idx=i,
                            model=xgb_model,
                            model_param_name="xgb_model",
                            model_score=xgb_model_score,
                        )

                        self.mlflow_op.log_all_for_model(
                            idx=i,
                            model=rf_model,
                            model_param_name="rf_model",
                            model_score=rf_model_score,
                        )

                except Exception as e:
                    self.log_writer.log(
                        table_name=self.model_train_log,
                        log_message="Mlflow logging of params,metrics and models failed",
                    )

                    self.log_writer.exception_log(
                        error=e,
                        class_name=self.class_name,
                        method_name=method_name,
                        table_name=self.model_train_log,
                    )

            self.log_writer.log(
                table_name=self.model_train_log,
                log_message="Successful End of Training",
            )

            return number_of_clusters

        except Exception as e:
            self.log_writer.log(
                table_name=self.model_train_log,
                log_message="Unsuccessful End of Training",
            )

            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=self.model_train_log,
            )
