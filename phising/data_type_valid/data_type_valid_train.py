from phising.mongo_db_operations.mongo_operations import MongoDB_Operation
from phising.s3_bucket_operations.s3_operations import S3_Operation
from utils.logger import App_Logger
from utils.read_params import read_params


class DB_Operation_Train:
    """
    Description :    This class shall be used for handling all the db operations

    Version     :    1.2
    Revisions   :    moved setup to cloud
    """

    def __init__(self):
        self.config = read_params()

        self.class_name = self.__class__.__name__

        self.train_data_bucket = self.config["bucket"]["phising_train_data"]

        self.train_export_csv_file = self.config["export_csv_file"]["train"]

        self.good_data_train_dir = self.config["data"]["train"]["good_data_dir"]

        self.input_files_bucket = self.config["bucket"]["input_files"]

        self.train_db_insert_log = self.config["train_db_log"]["db_insert"]

        self.train_export_csv_log = self.config["train_db_log"]["export_csv"]

        self.s3 = S3_Operation()

        self.mongo = MongoDB_Operation()

        self.log_writer = App_Logger()

    def insert_good_data_as_record(self, good_data_db_name, good_data_collection_name):
        """
        Method Name :   insert_good_data_as_record
        Description :   This method inserts the good data in MongoDB as collection

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.insert_good_data_as_record.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=self.train_db_insert_log,
        )

        try:
            lst = self.s3.read_csv_from_folder(
                folder_name=self.good_data_train_dir,
                bucket_name=self.train_data_bucket,
                table_name=self.train_db_insert_log,
            )

            for idx, f in enumerate(lst):
                df = f[idx][0]

                file = f[idx][1]

                if file.endswith(".csv"):
                    self.mongo.insert_dataframe_as_record(
                        data_frame=df,
                        db_name=good_data_db_name,
                        collection_name=good_data_collection_name,
                        table_name=self.train_db_insert_log,
                    )

                else:
                    pass

                self.log_writer.log(
                    table_name=self.train_db_insert_log,
                    log_message="Inserted dataframe as collection record in mongodb",
                )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=self.train_db_insert_log,
            )

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=self.train_db_insert_log,
            )

    def export_collection_to_csv(self, good_data_db_name, good_data_collection_name):
        """
        Method Name :   export_collection_to_csv

        Description :   This method extracts the inserted data to csv file, which will be used for training
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.export_collection_to_csv.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=self.train_export_csv_log,
        )

        try:
            df = self.mongo.get_collection_as_dataframe(
                db_name=good_data_db_name,
                collection_name=good_data_collection_name,
                table_name=self.train_export_csv_log,
            )

            self.s3.upload_df_as_csv(
                data_frame=df,
                local_file_name=self.train_export_csv_file,
                bucket_file_name=self.train_export_csv_file,
                bucket_name=self.input_files_bucket,
                table_name=self.input_files_bucket,
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=self.train_export_csv_log,
            )

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=self.train_export_csv_log,
            )
