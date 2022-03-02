from phising.s3_bucket_operations.S3_Operation import S3_Operation
from utils.logger import App_Logger
from utils.read_params import read_params


class data_getter_train:
    """
    Description :   This class shall be used for obtaining the df from the source for training
    Version     :   1.2
    Revisions   :   Moved to setup to cloud 
    """

    def __init__(self, table_name):
        self.config = read_params()

        self.table_name = table_name

        self.train_csv_file = self.config["export_csv_file"]["train"]

        self.input_files_bucket = self.config["s3_bucket"]["input_files_bucket"]

        self.s3 = S3_Operation()

        self.log_writer = App_Logger()

        self.class_name = self.__class__.__name__

    def get_data(self):
        """
        Method Name :   get_data
        Description :   This method reads the data from the source
        Output      :   A pandas dataframe
        On failure  :   Raise Exception
        Written by  :   iNeuron Intelligence
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_data.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=self.table_name,
        )

        try:
            df = self.s3.read_csv(
                bucket=self.input_files_bucket,
                file_name=self.train_csv_file,
                table_name=self.table_name,
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=self.table_name,
            )

            return df

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=self.table_name,
            )
