from phising.data_transform.data_transformation_pred import Data_Transform_Pred
from phising.data_type_valid.data_type_valid_pred import DB_Operation_Pred
from phising.raw_data_validation.pred_data_validation import raw_pred_data_validation
from utils.logger import App_Logger
from utils.read_params import read_params


class pred_validation:
    """
    Description :   This class is used for validating all the Prediction batch files

    Version     :   1.2
    Revisions   :   moved to setup to cloud
    """

    def __init__(self, bucket_name):
        self.raw_data = raw_pred_data_validation(raw_data_bucket_name=bucket_name)

        self.data_transform = Data_Transform_Pred()

        self.db_operation = DB_Operation_Pred()

        self.config = read_params()

        self.class_name = self.__class__.__name__

        self.db_name = self.config["db_log"]["db_pred_log"]

        self.pred_main_log = self.config["pred_db_log"]["pred_main"]

        self.good_data_db_name = self.config["mongodb"]["phising_data_db_name"]

        self.good_data_collection_name = self.config["mongodb"][
            "phising_pred_data_collection"
        ]

        self.log_writer = App_Logger()

    def prediction_validation(self):
        """
        Method Name :   load_s3
        Description :   This method is used for validating the Prediction btach files

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.prediction_validation.__name__

        try:
            self.log_writer.start_log(
                key="start",
                class_name=self.class_name,
                method_name=method_name,
                table_name=self.pred_main_log,
            )

            (
                LengthOfDateStampInFile,
                LengthOfTimeStampInFile,
                column_names,
                noofcolumns,
            ) = self.raw_data.values_from_schema()

            regex = self.raw_data.get_regex_pattern()

            self.raw_data.validate_raw_file_name(
                regex, LengthOfDateStampInFile, LengthOfTimeStampInFile
            )

            self.raw_data.validate_col_length(NumberofColumns=noofcolumns)

            self.raw_data.validate_missing_values_in_col()

            self.log_writer.log(
                table_name=self.pred_main_log,
                log_message="Raw Data Validation Completed !!",
            )

            self.log_writer.log(
                table_name=self.pred_main_log,
                log_message="Starting Data Transformation",
            )

            self.data_transform.rename_target_column()

            self.data_transform.replace_missing_with_null()

            self.log_writer.log(
                table_name=self.pred_main_log,
                log_message="Data Transformation completed !!",
            )

            self.db_operation.insert_good_data_as_record(
                db_name=self.good_data_db_name,
                collection_name=self.good_data_collection_name,
            )

            self.log_writer.log(
                table_name=self.pred_main_log,
                log_message="Data type validation Operation completed !!",
            )

            self.db_operation.export_collection_to_csv(
                db_name=self.good_data_db_name,
                collection_name=self.good_data_collection_name,
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=self.pred_main_log,
            )

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=self.pred_main_log,
            )
