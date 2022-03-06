from phising.s3_bucket_operations.s3_operations import S3_Operation
from utils.logger import App_Logger
from utils.read_params import read_params


class Data_Getter_Pred:
    """
    Description :   This class shall be used for obtaining the df from the input files s3 bucket where the prediction file is present
    Written by  :   iNeuron Intelligence
    
    Version     :   1.2
    Revisions   :   Moved to setup to cloud 
    """

    def __init__(self, table_name):
        self.config = read_params()

        self.table_name = table_name

        self.prediction_file = self.config["export_csv_file"]["pred"]

        self.input_files_bucket = self.config["s3_bucket"]["input_files"]

        self.s3 = S3_Operation()

        self.log_writer = App_Logger()

        self.class_name = self.__class__.__name__

    def get_data(self):
        """
        Method Name :   get_data
        Description :   This method reads the data from the input files s3 bucket where the prediction file is present
        Output      :   A pandas dataframe
        
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Written by  :   iNeuron Intelligence
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
                file_name=self.prediction_file,
                bucket_name=self.input_files_bucket,
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
