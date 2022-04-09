from phising.s3_bucket_operations.s3_operations import S3_Operation
from utils.logger import App_Logger
from utils.read_params import read_params


class Data_Transform_Pred:
    """
    Description :  This class shall be used for transforming the Prediction batch data before loading it in Database!!.
    Written by  :   iNeuron Intelligence
    
    Version     :   1.2
    Revisions   :   Moved to setup to cloud 
    """

    def __init__(self):
        self.config = read_params()

        self.pred_data_bucket = self.config["s3_bucket"]["phising_pred_data"]

        self.s3 = S3_Operation()

        self.log_writer = App_Logger()

        self.good_pred_data_dir = self.config["data"]["pred"]["good"]

        self.class_name = self.__class__.__name__

        self.db_name = self.config["db_log"]["pred"]

        self.pred_data_transform_log = self.config["pred_db_log"]["data_transform"]

    def add_quotes_to_string(self):
        """
        Method Name :   add_quotes_to_string
        Description :   This method addes the quotes to the string data present in columns

        Output      :   A csv file where all the string values have quotes inserted
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Written by  :   iNeuron Intelligence
        Revisions   :   moved setup to cloud
        """
        method_name = self.add_quotes_to_string.__name__

        self.log_writer.start_log(
            "start", self.class_name, method_name, self.pred_data_transform_log,
        )

        try:
            lst = self.s3.read_csv_from_folder(
                self.good_pred_data_dir,
                self.pred_data_bucket,
                self.pred_data_transform_log,
            )

            for idx, f in enumerate(lst):
                df = f[idx][0]

                file = f[idx][1]

                abs_f = f[idx][2]

                if file.endswith(".csv"):
                    for column in df.columns:
                        count = df[column][df[column] == "?"].count()

                        if count != 0:
                            df[column] = df[column].replace("?", "'?'")

                    self.log_writer.log(
                        f"Quotes added for the file {file}",
                        self.pred_data_transform_log,
                    )

                    self.s3.upload_df_as_csv(
                        df,
                        abs_f,
                        file,
                        self.pred_data_bucket,
                        self.pred_data_transform_log,
                    )

                else:
                    pass

            self.log_writer.start_log(
                "exit", self.class_name, method_name, self.pred_data_transform_log,
            )

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name, self.pred_data_transform_log,
            )
