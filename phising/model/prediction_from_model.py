import pandas as pd
from botocore.exceptions import ClientError
from phising.data_ingestion.data_loader_prediction import Data_Getter_Pred
from phising.data_preprocessing.preprocessing import Preprocessor
from phising.s3_bucket_operations.s3_operations import S3_Operation
from utils.logger import App_Logger
from utils.read_params import read_params


class Prediction:
    """
    Description :   This class shall be used for loading the production model
    Written by  :   iNeuron Intelligence
    
    Version     :   1.2
    Revisions   :   Moved to setup to cloud 
    """

    def __init__(self):
        self.config = read_params()

        self.pred_log = self.config["pred_db_log"]["pred_main"]

        self.model_bucket_name = self.config["s3_bucket"]["phising_model"]

        self.input_files_bucket = self.config["s3_bucket"]["inputs_files"]

        self.prod_model_dir = self.config["models_dir"]["prod"]

        self.pred_output_file = self.config["pred_output_file"]

        self.log_writer = App_Logger()

        self.s3 = S3_Operation()

        self.data_getter_pred = Data_Getter_Pred(self.pred_log)

        self.preprocessor = Preprocessor(self.pred_log)

        self.class_name = self.__class__.__name__

    def delete_pred_file(self, log_file):
        """
        Method Name :   delete_pred_file
        Description :   This method deletes the existing prediction file for the model prediction starts

        Output      :   An existing prediction file is deleted
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Written by  :   iNeuron Intelligence
        Revisions   :   moved setup to cloud
        """
        method_name = self.delete_pred_file.__name__

        self.log_writer.start_log(
            "start",
            self.class_name,
            method_name,
            
        )

        try:
            self.s3.load_object(
                object=self.pred_output_file,
                self.input_files_bucket,
                
            )

            self.log_writer.log(
                
                log_file,f"Found existing Prediction batch file. Deleting it.",
            )

            self.s3.delete_file(
                file_name=self.pred_output_file,
                self.input_files_bucket,
                
            )

            self.log_writer.start_log(
                "exit",
                self.class_name,
                method_name,
                
            )

        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                pass

            else:
                self.log_writer.exception_log(
                    e,
                    self.class_name,
                    method_name,
                    
                )

    def find_correct_model_file(self, cluster_number, bucket, log_file):
        """
        Method Name :   find_correct_model_file
        Description :   This method gets correct model file based on cluster number during prediction

        Output      :   A correct model file is found 
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Written by  :   iNeuron Intelligence
        Revisions   :   moved setup to cloud
        """
        method_name = self.find_correct_model_file.__name__

        self.log_writer.start_log(
            "start",
            self.class_name,
            method_name,
            
        )

        try:
            list_of_files = self.s3.get_files_from_folder(
                bucket=bucket,
                self.prod_model_dir,
                
            )

            for file in list_of_files:
                try:
                    if file.index(str(cluster_number)) != -1:
                        model_name = file

                except:
                    continue

            model_name = model_name.split(".")[0]

            self.log_writer.log(
                
                log_file,f"Got {model_name} from {self.prod_model_dir} folder in {bucket} bucket",
            )

            self.log_writer.start_log(
                "exit",
                self.class_name,
                method_name,
                
            )

            return model_name

        except Exception as e:
            self.log_writer.exception_log(
                e,
                self.class_name,
                method_name,
                
            )

    def predict_from_model(self):
        """
        Method Name :   predict_from_model
        Description :   This method predicts the new data using the existing models

        Output      :   Prediction file is created in input files bucket
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Written by  :   iNeuron Intelligence
        Revisions   :   moved setup to cloud
        """
        method_name = self.predict_from_model.__name__

        self.log_writer.start_log(
            "start",
            self.class_name,
            method_name,
            self.pred_log,
        )

        try:
            self.delete_pred_file(self.pred_log)

            data = self.data_getter_pred.get_data()

            data = self.preprocessor.replace_invalid_values(data)

            is_null_present = self.preprocessor.is_null_present(data)

            if is_null_present:
                data = self.preprocessor.impute_missing_values(data)

            kmeans = self.s3.load_model(
                model_name="KMeans",
                self.model_bucket_name,
                self.pred_log,
            )

            clusters = kmeans.predict(data.drop(["clusters"], axis=1))

            data["clusters"] = clusters

            clusters = data["clusters"].unique()

            for i in clusters:
                cluster_data = data[data["clusters"] == i]

                phising_names = list(cluster_data["phising"])

                cluster_data = data.drop(labels=["phising"], axis=1)

                cluster_data = cluster_data.drop(["clusters"], axis=1)

                crt_model_name = self.find_correct_model_file(
                    cluster_number=i,
                    self.model_bucket_name,
                    self.pred_log,
                )

                model = self.s3.load_model(model_name=crt_model_name)

                result = list(model.predict(cluster_data))

                result = pd.DataFrame(
                    list(zip(phising_names, result)), columns=["phising", "prediction"]
                )

                self.s3.upload_df_as_csv(
                    result,
                    self.pred_output_file,
                    self.input_files_bucket,
                    self.input_files_bucket,
                    self.pred_log,
                )

            self.log_writer.log(self.pred_log, log_file,"End of Prediction")

            self.log_writer.start_log(
                "exit",
                self.class_name,
                method_name,
                self.pred_log,
            )

            return (
                self.input_files_bucket,
                self.pred_output_file,
                result.head().to_json(orient="records"),
            )

        except Exception as e:
            self.log_writer.exception_log(
                e,
                self.class_name,
                method_name,
                self.pred_log,
            )
