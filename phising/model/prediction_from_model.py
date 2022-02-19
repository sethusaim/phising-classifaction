import pandas as pd
from phising.data_ingestion.data_loader_prediction import data_getter_pred
from phising.data_preprocessing.preprocessing import preprocessor
from phising.s3_bucket_operations.s3_operations import s3_operations
from utils.logger import app_logger
from utils.read_params import read_params


class prediction:
    """
    Description :   This class shall be used for loading the production model

    Version     :   1.2
    Revisions   :   moved to setup to cloud
    """

    def __init__(self):
        self.config = read_params()

        self.pred_log = self.config["pred_db_log"]["pred_main"]

        self.model_bucket = self.config["s3_bucket"]["phising_model_bucket"]

        self.input_files_bucket = self.config["s3_bucket"]["inputs_files_bucket"]

        self.prod_model_dir = self.config["models_dir"]["prod"]

        self.pred_output_file = self.config["pred_output_file"]

        self.log_writer = app_logger()

        self.s3 = s3_operations()

        self.data_getter_pred = data_getter_pred(table_name=self.pred_log)

        self.preprocessor = preprocessor(table_name=self.pred_log)

        self.class_name = self.__class__.__name__

    def predict_from_model(self):
        """
        Method Name :   predict_from_model
        Description :   This method is used for loading from prod model dir of s3 bucket and use them for prediction

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.predict_from_model.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=self.pred_log,
        )

        try:
            self.s3.delete_pred_file(table_name=self.pred_log)

            data = self.data_getter_pred.get_data()

            data = self.preprocessor.replace_invalid_values(data=data)

            is_null_present = self.preprocessor.is_null_present(data=data)

            if is_null_present:
                data = self.preprocessor.impute_missing_values(data=data)

            cols_to_drop = self.preprocessor.get_columns_with_zero_std_deviation(
                data=data
            )

            X = self.preprocessor.remove_columns(data, cols_to_drop)

            X = self.preprocessor.scale_numerical_columns(data=X)

            X = self.preprocessor.apply_pca_transform(X_scaled_data=X)

            kmeans_model_name = self.prod_model_dir + "/" + "KMeans"

            kmeans_model = self.s3.load_model(
                bucket=self.model_bucket,
                model_name=kmeans_model_name,
                table_name=self.pred_log,
            )

            clusters = kmeans_model.predict(data)

            data["clusters"] = clusters

            unique_clusters = data["clusters"].unique()

            for i in unique_clusters:
                cluster_data = data[data["clusters"] == i]

                cluster_data = cluster_data.drop(["clusters"], axis=1)

                model_name = self.s3.find_correct_model_file(
                    cluster_number=i,
                    bucket_name=self.model_bucket,
                    table_name=self.pred_log,
                )

                prod_model_name = self.prod_model_dir + "/" + model_name

                model = self.s3.load_model(
                    bucket=self.model_bucket,
                    model_name=prod_model_name,
                    table_name=self.pred_log,
                )

                result = list(model.predict(cluster_data))

                result = pd.DataFrame(result, columns=["predictions"])

                result["predictions"] = result["predictions"].map({0: "neg", 1: "pos"})

                self.s3.upload_df_as_csv(
                    data_frame=result,
                    file_name=self.pred_output_file,
                    bucket=self.input_files_bucket,
                    dest_file_name=self.pred_output_file,
                    table_name=self.pred_log,
                )

            self.log_writer.log(
                table_name=self.pred_log, log_message="End of prediction"
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=self.pred_log,
            )

            return (
                self.input_files_bucket,
                self.pred_output_file,
                result.head().to_json(orient="records"),
            )

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=self.pred_log,
            )
