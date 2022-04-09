from re import match, split

from phising.s3_bucket_operations.s3_operations import S3_Operation
from utils.logger import App_Logger
from utils.read_params import read_params


class Raw_Train_Data_Validation:
    """
    Description :   This method is used for validating the raw training data
    Written by  :   iNeuron Intelligence
    
    Version     :   1.2
    Revisions   :   Moved to setup to cloud 
    """

    def __init__(self, raw_data_bucket: str):
        self.config = read_params()

        self.raw_data_bucket = raw_data_bucket

        self.log_writer = App_Logger()

        self.class_name = self.__class__.__name__

        self.s3 = S3_Operation()

        self.train_data_bucket = self.config["s3_bucket"]["phising_train_data"]

        self.input_files_bucket = self.config["s3_bucket"]["input_files"]

        self.raw_train_data_dir = self.config["data"]["raw_data"]["train_batch"]

        self.train_schema_file = self.config["schema_file"]["train"]

        self.regex_file = self.config["regex_file"]

        self.train_schema_log = self.config["train_db_log"]["values_from_schema"]

        self.good_train_data_dir = self.config["data"]["train"]["good"]

        self.bad_train_data_dir = self.config["data"]["train"]["bad"]

        self.train_gen_log = self.config["train_db_log"]["general"]

        self.train_name_valid_log = self.config["train_db_log"]["name_validation"]

        self.train_col_valid_log = self.config["train_db_log"]["col_validation"]

        self.train_missing_value_log = self.config["train_db_log"][
            "missing_values_in_col"
        ]

    def values_from_schema(self):
        """
        Method Name :   values_from_schema
        Description :   This method gets schema values from the schema_training.json file

        Output      :   Schema values are extracted from the schema_training.json file
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.values_from_schema.__name__

        try:
            self.log_writer.start_log(
                "start", self.class_name, method_name, self.train_schema_log,
            )

            dic = self.s3.read_json(
                self.train_schema_file, self.input_files_bucket, self.train_schema_log,
            )

            LengthOfDateStampInFile = dic["LengthOfDateStampInFile"]

            LengthOfTimeStampInFile = dic["LengthOfTimeStampInFile"]

            column_names = dic["ColName"]

            NumberofColumns = dic["NumberofColumns"]

            message = (
                "LengthOfDateStampInFile:: %s" % LengthOfDateStampInFile
                + "\t"
                + "LengthOfTimeStampInFile:: %s" % LengthOfTimeStampInFile
                + "\t "
                + "NumberofColumns:: %s" % NumberofColumns
                + "\n"
            )

            self.log_writer.log(
                self.train_schema_log, message,
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name, self.train_schema_log,
            )

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name, self.train_schema_log,
            )

        return (
            LengthOfDateStampInFile,
            LengthOfTimeStampInFile,
            column_names,
            NumberofColumns,
        )

    def get_regex_pattern(self):
        """
        Method Name :   get_regex_pattern
        Description :   This method gets regex pattern from input files s3 bucket

        Output      :   A regex pattern is extracted
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_regex_pattern.__name__

        try:
            self.log_writer.start_log(
                "start", self.class_name, method_name, self.train_gen_log,
            )

            regex = self.s3.read_text(
                self.regex_file, self.input_files_bucket, self.train_gen_log,
            )

            self.log_writer.log(f"Got {regex} pattern", self.train_gen_log)

            self.log_writer.start_log(
                "exit", self.class_name, method_name, self.train_gen_log,
            )

            return regex

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name, self.train_gen_log,
            )

    def create_dirs_for_good_bad_data(self, log_file):
        """
        Method Name :   create_dirs_for_good_bad_data
        Description :   This method creates folders for good and bad data in s3 bucket

        Output      :   Good and bad folders are created in s3 bucket
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.create_dirs_for_good_bad_data.__name__

        self.log_writer.start_log("start", self.class_name, method_name, log_file)

        try:
            self.s3.create_folder(
                self.good_train_data_dir, self.train_data_bucket, log_file
            )

            self.s3.create_folder(
                self.bad_train_data_dir, self.train_data_bucket, log_file
            )

            self.log_writer.start_log("exit", self.class_name, method_name, log_file)

        except Exception as e:
            self.log_writer.exception_log(e, self.class_name, method_name, log_file)

    def validate_raw_file_name(
        self, regex: str, LengthOfDateStampInFile: int, LengthOfTimeStampInFile: int
    ):
        """
        Method Name :   validate_raw_file_name
        Description :   This method validates the raw file name based on regex pattern and schema values

        Output      :   Raw file names are validated, good file names are stored in good data folder and rest is stored in bad data
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.validate_raw_file_name.__name__

        self.log_writer.start_log(
            "start", self.class_name, method_name, self.train_name_valid_log,
        )

        try:
            self.create_dirs_for_good_bad_data(self.train_name_valid_log)

            onlyfiles = self.s3.get_files_from_folder(
                self.raw_data_bucket,
                self.raw_train_data_dir,
                self.train_name_valid_log,
            )

            train_batch_files = [f.split("/")[1] for f in onlyfiles]

            self.log_writer.log(
                "Got training files with absolute file name", self.train_name_valid_log
            )

            for fname in train_batch_files:
                raw_data_train_file_name = self.raw_train_data_dir + "/" + fname

                good_data_train_file_name = self.good_train_data_dir + "/" + fname

                bad_data_train_file_name = self.bad_train_data_dir + "/" + fname

                self.log_writer.log(
                    "Created raw,good and bad data file name", self.train_name_valid_log
                )

                if match(regex, fname):
                    splitAtDot = split(".csv", fname)

                    splitAtDot = split("_", splitAtDot[0])

                    if len(splitAtDot[1]) == LengthOfDateStampInFile:
                        if len(splitAtDot[2]) == LengthOfTimeStampInFile:
                            self.s3.copy_data(
                                raw_data_train_file_name,
                                self.train_data_bucket,
                                good_data_train_file_name,
                                self.train_data_bucket,
                                self.train_name_valid_log,
                            )

                        else:
                            self.s3.copy_data(
                                raw_data_train_file_name,
                                self.train_data_bucket,
                                bad_data_train_file_name,
                                self.train_data_bucket,
                                self.train_name_valid_log,
                            )

                    else:
                        self.s3.copy_data(
                            raw_data_train_file_name,
                            self.train_data_bucket,
                            bad_data_train_file_name,
                            self.train_data_bucket,
                            self.train_name_valid_log,
                        )
                else:
                    self.s3.copy_data(
                        raw_data_train_file_name,
                        self.train_data_bucket,
                        bad_data_train_file_name,
                        self.train_data_bucket,
                        self.train_name_valid_log,
                    )

            self.log_writer.start_log(
                "exit", self.class_name, method_name, self.train_name_valid_log,
            )

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name, self.train_name_valid_log,
            )

    def validate_col_length(self, NumberofColumns: int):
        """
        Method Name :   validate_col_length
        Description :   This method validates the column length based on number of columns as mentioned in schema values

        Output      :   The files' columns length are validated and good data is stored in good data folder and rest is stored in bad data folder
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.validate_col_length.__name__

        self.log_writer.start_log(
            "start", self.class_name, method_name, self.train_col_valid_log,
        )

        try:
            lst = self.s3.read_csv_from_folder(
                self.good_train_data_dir,
                self.train_data_bucket,
                self.train_col_valid_log,
            )

            for idx, f in enumerate(lst):
                df = f[idx][0]

                file = f[idx][1]

                abs_f = f[idx][2]

                if file.endswith(".csv"):
                    if df.shape[1] == NumberofColumns:
                        pass

                    else:
                        dest_f = self.bad_train_data_dir + "/" + abs_f

                        self.s3.move_data(
                            file,
                            self.train_data_bucket,
                            dest_f,
                            self.train_data_bucket,
                            self.train_col_valid_log,
                        )

                else:
                    pass

            self.log_writer.start_log(
                "exit", self.class_name, method_name, self.train_col_valid_log,
            )

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name, self.train_col_valid_log,
            )

    def validate_missing_values_in_col(self):
        """
        Method Name :   validate_missing_values_in_col
        Description :   This method validates the missing values in columns

        Output      :   Missing columns are validated, and good data is stored in good data folder and rest is to stored in bad data folder
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.validate_missing_values_in_col.__name__

        self.log_writer.start_log(
            "start", self.class_name, method_name, self.train_missing_value_log,
        )

        try:
            lst = self.s3.read_csv_from_folder(
                self.good_train_data_dir,
                self.train_data_bucket,
                self.train_missing_value_log,
            )

            for idx, f in lst:
                df = f[idx][0]

                file = f[idx][1]

                abs_f = f[idx][2]

                if abs_f.endswith(".csv"):
                    count = 0

                    for cols in df:
                        if (len(df[cols]) - df[cols].count()) == len(df[cols]):
                            count += 1

                            dest_f = self.bad_train_data_dir + "/" + abs_f

                            self.s3.move_data(
                                file,
                                self.train_data_bucket,
                                dest_f,
                                self.train_data_bucket,
                                self.train_missing_value_log,
                            )

                            break

                    if count == 0:
                        dest_f = self.good_train_data_dir + "/" + abs_f

                        self.s3.upload_df_as_csv(
                            df,
                            abs_f,
                            dest_f,
                            self.train_data_bucket,
                            self.train_missing_value_log,
                        )

                else:
                    pass

                self.log_writer.start_log(
                    "exit", self.class_name, method_name, self.train_missing_value_log,
                )

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name, self.train_missing_value_log,
            )
