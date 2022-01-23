from phising.s3_bucket_operations.s3_operations import S3_Operations
from utils.logger import App_Logger
from utils.main_utils import convert_object_to_dataframe
from utils.read_params import read_params


class data_getter_train:
    """
    Description :   This class shall be used for obtaining the df from the source for prediction
    Version     :   1.2
    Revisions   :   Moved to setup to cloud run setup
    """

    def __init__(self, db_name, collection_name):
        self.config = read_params()

        self.training_file = self.config["export_train_csv_file"]

        self.db_name = db_name

        self.input_files_bucket = self.config["s3_bucket"]["input_files_bucket"]

        self.s3_obj = S3_Operations()

        self.collection_name = collection_name

        self.log_writer = App_Logger()

        self.class_name = self.__class__.__name__

    def get_data(self):
        """
        Method Name :   get_data
        Description :   This method reads the data from the source
        Output      :   A pandas dataframe
        On failure  :   Raise Exception
        Version     :   1.2
        Revisions   :   Moved to setup to cloud run setup
        """
        method_name = self.get_data.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            db_name=self.db_name,
            collection_name=self.collection_name,
        )

        try:
            csv_obj = self.s3_obj.get_file_objects_from_s3(
                bucket=self.input_files_bucket,
                filename=self.training_file,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            df = convert_object_to_dataframe(
                obj=csv_obj, db_name=self.db_name, collection_name=self.collection_name
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            return df

        except Exception as e:
            self.log_writer.raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )
