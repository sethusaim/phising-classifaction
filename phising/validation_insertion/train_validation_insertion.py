from phising.data_transform.data_transformation_train import Data_Transform_Train
from phising.data_type_valid.data_type_valid_train import DB_Operation_Train
from phising.raw_data_validation.train_data_validation import Raw_Train_Data_Validation
from utils.logger import App_Logger
from utils.read_params import read_params


class Train_Validation:
    """
    Description :   This class is used for validating all the training batch files

    Version     :   1.2
    Revisions   :   moved to setup to cloud
    """

    def __init__(self, bucket_name):
        self.raw_data = Raw_Train_Data_Validation(raw_data_bucket_name=bucket_name)

        self.data_transform = Data_Transform_Train()

        self.db_operation = DB_Operation_Train()

        self.config = read_params()

        self.class_name = self.__class__.__name__

        self.db_name = self.config["db_log"]["db_train_log"]

        self.train_main_log = self.config["train_db_log"]["train_main"]

        self.good_data_db_name = self.config["mongodb"]["phising_data_db_name"]

        self.good_data_collection_name = self.config["mongodb"][
            "phising_train_data_collection"
        ]

        self.log_writer = App_Logger()

    def train_validation(self):
        """
        Method Name :   train_validation
        Description :   This method is used for validating the training batch files

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.train_validation.__name__

        try:
            self.log_writer.start_log(
                key="start",
                class_name=self.class_name,
                method_name=method_name,
                table_name=self.train_main_log,
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
                table_name=self.train_main_log,
                log_message="Raw Data Validation Completed !!",
            )

            self.log_writer.log(
                table_name=self.train_main_log,
                log_message="Starting Data Transformation",
            )

            self.data_transform.add_quotes_to_string()

            self.log_writer.log(
                table_name=self.train_main_log,
                log_message="Data Transformation completed !!",
            )

            self.db_operation.insert_good_data_as_record(
                db_name=self.good_data_db_name,
                table_name=self.good_data_collection_name,
            )

            self.log_writer.log(
                table_name=self.train_main_log,
                log_message="Data type validation Operation completed !!",
            )

            self.db_operation.export_collection_to_csv(
                db_name=self.good_data_db_name,
                table_name=self.good_data_collection_name,
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=self.train_main_log,
            )

        except Exception as e:
            self.log_writer.raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=self.train_main_log,
            )
