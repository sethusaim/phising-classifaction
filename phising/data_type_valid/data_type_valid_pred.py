from phising.mongo_db_operations.mongo_operations import MongoDB_Operation
from phising.s3_bucket_operations.s3_operations import S3_Operations
from utils.logger import App_Logger
from utils.main_utils import convert_object_to_dataframe
from utils.read_params import read_params


class db_operation_pred:
    """
    Description :    This class shall be used for handling all the db operations

    Version     :    1.2
    Revisions   :    moved setup to cloud
    """

    def __init__(self):
        self.config = read_params()

        self.s3_obj = S3_Operations()

        self.db_op = MongoDB_Operation()

        self.log_writer = App_Logger()

        self.class_name = self.__class__.__name__

        self.pred_data_bucket = self.config["s3_bucket"]["phising_pred_data_bucket"]

        self.pred_export_csv_file = self.config["export_pred_csv_file"]

        self.good_data_pred_dir = self.config["data"]["pred"]["good_data_dir"]

        self.input_files_bucket = self.config["s3_bucket"]["input_files_bucket"]

        self.db_name = self.config["db_log"]["db_pred_log"]

        self.pred_db_insert_log = self.config["pred_db_log"]["db_insert"]

        self.pred_export_csv_log = self.config["pred_db_log"]["export_csv"]

    def insert_good_data_as_record(self, db_name, collection_name):
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
            db_name=self.db_name,
            collection_name=self.pred_db_insert_log,
        )

        try:
            csv_files = self.s3_obj.get_file_objects_from_s3(
                bucket=self.pred_data_bucket,
                filename=self.good_data_pred_dir,
                db_name=self.db_name,
                collection_name=self.pred_db_insert_log,
            )

            for f in csv_files:
                file = f.key

                if file.endswith(".csv"):
                    df = convert_object_to_dataframe(
                        obj=f,
                        db_name=self.db_name,
                        collection_name=self.pred_db_insert_log,
                    )

                    self.db_op.insert_dataframe_as_record(
                        data_frame=df, db_name=db_name, collection_name=collection_name,
                    )

                else:
                    pass

                self.log_writer.log(
                    db_name=self.db_name,
                    collection_name=self.pred_db_insert_log,
                    log_message="Inserted dataframe as collection record in mongodb",
                )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.pred_db_insert_log,
            )

        except Exception as e:
            self.log_writer.raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.pred_db_insert_log,
            )

    def export_collection_to_csv(self, db_name, collection_name):
        """
        Method Name :   export_collection_to_csv

        Description :   This method extracts the inserted data to csv file, which will be used for preding
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.export_collection_to_csv.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            db_name=self.db_name,
            collection_name=self.pred_export_csv_log,
        )

        try:
            df = self.db_op.convert_collection_to_dataframe(
                db_name=db_name, collection_name=collection_name
            )

            self.s3_obj.upload_df_as_csv_to_s3(
                data_frame=df,
                file_name=self.pred_export_csv_file,
                bucket=self.input_files_bucket,
                dest_file_name=self.pred_export_csv_file,
                db_name=self.db_name,
                collection_name=self.pred_export_csv_log,
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.pred_export_csv_log,
            )

        except Exception as e:
            self.log_writer.raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=db_name,
                collection_name=collection_name,
            )
