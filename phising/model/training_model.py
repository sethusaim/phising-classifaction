import mlflow
from phising.data_ingestion.data_loader_train import data_getter_train
from phising.data_preprocessing.clustering import kmeans_clustering
from phising.data_preprocessing.preprocessing import preprocessor
from phising.mlflow_utils.mlflow_operations import mlflow_operations
from phising.model_finder.tuner import model_finder
from phising.s3_bucket_operations.s3_operations import s3_operations
from sklearn.model_selection import train_test_split
from utils.logger import app_logger
from utils.model_utils import get_model_name
from utils.read_params import read_params


class train_model:
    """
    Description :   This method is used for getting the data and applying
                    some preprocessing steps and then train the models and register them in mlflow

    Version     :   1.2
    Revisions   :   moved to setup to cloud
    """

    def __init__(self):
        self.log_writer = app_logger()

        self.config = read_params()

        self.model_train_log = self.config["train_db_log"]["model_training"]

        self.model_bucket = self.config["s3_bucket"]["phising_model_bucket"]

        self.test_size = self.config["base"]["test_size"]

        self.target_col = self.config["base"]["target_col"]

        self.random_state = self.config["base"]["random_state"]

        self.experiment_name = self.config["mlflow_config"]["experiment_name"]

        self.run_name = self.config["mlflow_config"]["run_name"]

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
            df = self.data_getter_train_obj.get_data()

            df = self.preprocessor_obj.replace_invalid_values(data=df)

            df = self.preprocessor_obj.encode_target_cols(data=df)

            is_null_present = self.preprocessor_obj.is_null_present(data=df)

            if is_null_present:
                df = self.preprocessor_obj.impute_missing_values(data=df)

            X, Y = self.preprocessor_obj.separate_label_feature(
                data=df, label_column_name=self.target_col
            )

            cols_to_drop = self.preprocessor_obj.get_columns_with_zero_std_deviation(
                data=X
            )

            X = self.preprocessor_obj.remove_columns(data=X, columns=cols_to_drop)

            X = self.preprocessor_obj.scale_numerical_columns(data=X)

            X = self.preprocessor_obj.apply_pca_transform(X_scaled_data=X)

            number_of_clusters = self.kmeans_obj.elbow_plot(data=X)

            X, kmeans_model = self.kmeans_obj.create_clusters(
                data=X, number_of_clusters=number_of_clusters
            )

            X["Labels"] = Y

            list_of_clusters = X["Cluster"].unique()

            self.log_writer.log(
                table_name=self.model_train_log,
                log_message="Got unique list of clusters",
            )

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
                    rf_model,
                    rf_model_score,
                    ada_model,
                    ada_model_score,
                ) = self.model_finder_obj.get_trained_models(
                    train_x=x_train, train_y=y_train, test_x=x_test, test_y=y_test
                )

                self.s3.save_model_to_s3(
                    idx=i,
                    model=ada_model,
                    model_bucket=self.model_bucket,
                    table_name=self.model_train_log,
                )

                self.s3.save_model_to_s3(
                    idx=i,
                    model=rf_model,
                    model_bucket=self.model_bucket,
                    table_name=self.model_train_log,
                )

                try:
                    self.mlflow_op.set_mlflow_tracking_uri()

                    self.mlflow_op.set_mlflow_experiment(
                        experiment_name=self.experiment_name
                    )

                    with mlflow.start_run(run_name=self.run_name):
                        kmeans_model_name = get_model_name(
                            model=kmeans_model, table_name=self.model_train_log
                        )

                        self.mlflow_op.log_model(
                            model=kmeans_model, model_name=kmeans_model_name
                        )

                        self.mlflow_op.log_all_for_model(
                            idx=i,
                            model=ada_model,
                            model_param_name="adaboost_model",
                            model_score=ada_model_score,
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

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=self.model_train_log,
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
