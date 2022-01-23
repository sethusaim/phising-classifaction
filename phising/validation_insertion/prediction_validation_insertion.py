import re

from phising.s3_bucket_operations.s3_operations import S3_Operations
from utils.logger import App_Logger
from utils.main_utils import convert_object_to_dataframe
from utils.read_params import read_params


class raw_train_data_validation:
    """
    Description :   This method is used for validating the raw training data

    Version     :   1.2
    Revisions   :   moved to setup to cloud
    """

    def __init__(self, raw_data_bucket_name):
        self.config = read_params()

        self.raw_data_bucket_name = raw_data_bucket_name

        self.log_writer = App_Logger()

        self.class_name = self.__class__.__name__

        self.s3_obj = S3_Operations()

        self.train_data_bucket = self.config["s3_bucket"]["phising_train_data_bucket"]

        self.input_files_bucket = self.config["s3_bucket"]["input_files_bucket"]

        self.raw_train_data_dir = self.config["data"]["raw_data"]["train_batch"]

        self.db_name = self.config["db_log"]["db_train_log"]

        self.train_schema_file = self.config["schema_file"]["train_schema_file"]

        self.train_schema_log = self.config["train_db_log"]["values_from_schema"]

        self.good_train_data_dir = self.config["data"]["train"]["good_data_dir"]

        self.bad_train_data_dir = self.config["data"]["train"]["bad_data_dir"]

        self.train_gen_log = self.config["train_db_log"]["general"]

        self.train_name_valid_log = self.config["train_db_log"]["name_validation"]

        self.train_col_valid_log = self.config["train_db_log"]["col_validation"]

        self.train_missing_value_log = self.config["train_db_log"][
            "missing_values_in_col"
        ]

    def values_from_schema(self):
        """
        Method Name :   values_from_schema
        Description :   This method is used for getting values from schema_training.json

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.values_from_schema.__name__

        try:
            self.log_writer.start_log(
                key="start",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.train_schema_log,
            )

            dic = self.s3_obj.get_schema_from_s3(
                bucket=self.input_files_bucket,
                filename=self.train_schema_file,
                db_name=self.db_name,
                collection_name=self.train_schema_log,
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
                db_name=self.db_name,
                collection_name=self.train_schema_log,
                log_message=message,
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.train_schema_log,
            )

            return (
                LengthOfDateStampInFile,
                LengthOfTimeStampInFile,
                column_names,
                NumberofColumns,
            )

        except Exception as e:
            self.log_writer.raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.train_schema_log,
            )

    def get_regex_pattern(self):
        """
        Method Name :   get_regex_pattern
        Description :   This method is used for getting regex pattern for file validation

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_regex_pattern.__name__

        try:
            self.log_writer.start_log(
                key="start",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.train_gen_log,
            )

            regex = "['phising']+['\_'']+[\d_]+[\d]+\.csv"

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.train_gen_log,
                log_message=f"Got {regex} pattern",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.train_gen_log,
            )

            return regex

        except Exception as e:
            self.log_writer.raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.train_gen_log,
            )

    def validate_raw_file_name(
        self, regex, LengthOfDateStampInFile, LengthOfTimeStampInFile
    ):
        """
        Method Name :   validate_raw_file_name
        Description :   This method is used for validating raw file name based on the regex pattern

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.validate_raw_file_name.__name__

        try:
            self.log_writer.start_log(
                key="start",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.train_name_valid_log,
            )

            self.s3_obj.create_dirs_for_good_bad_data(
                db_name=self.db_name, collection_name=self.train_name_valid_log
            )

            onlyfiles = self.s3_obj.get_files_from_s3(
                bucket=self.raw_data_bucket_name,
                folder_name=self.raw_train_data_dir,
                db_name=self.db_name,
                collection_name=self.train_name_valid_log,
            )

            train_batch_files = [f.split("/")[1] for f in onlyfiles]

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.train_name_valid_log,
                log_message="Got train batch files without path",
            )

            for filename in train_batch_files:
                raw_data_train_filename = self.raw_train_data_dir + "/" + filename

                good_data_train_filename = self.good_train_data_dir + "/" + filename

                bad_data_train_filename = self.bad_train_data_dir + "/" + filename

                if re.match(regex, filename):
                    splitAtDot = re.split(".csv", filename)

                    splitAtDot = re.split("_", splitAtDot[0])

                    if len(splitAtDot[1]) == LengthOfDateStampInFile:
                        if len(splitAtDot[2]) == LengthOfTimeStampInFile:
                            self.s3_obj.copy_data_to_other_bucket(
                                src_bucket=self.raw_data_bucket_name,
                                src_file=raw_data_train_filename,
                                dest_bucket=self.train_data_bucket,
                                dest_file=good_data_train_filename,
                                db_name=self.db_name,
                                collection_name=self.train_name_valid_log,
                            )

                        else:
                            self.s3_obj.copy_data_to_other_bucket(
                                src_bucket=self.raw_data_bucket_name,
                                src_file=raw_data_train_filename,
                                dest_bucket=self.train_data_bucket,
                                dest_file=bad_data_train_filename,
                                db_name=self.db_name,
                                collection_name=self.train_name_valid_log,
                            )

                    else:
                        self.s3_obj.copy_data_to_other_bucket(
                            src_bucket=self.raw_data_bucket_name,
                            src_file=raw_data_train_filename,
                            dest_bucket=self.train_data_bucket,
                            dest_file=bad_data_train_filename,
                            db_name=self.db_name,
                            collection_name=self.train_name_valid_log,
                        )

                else:
                    self.s3_obj.copy_data_to_other_bucket(
                        src_bucket=self.raw_data_bucket_name,
                        src_file=raw_data_train_filename,
                        dest_bucket=self.train_data_bucket,
                        dest_file=bad_data_train_filename,
                        db_name=self.db_name,
                        collection_name=self.train_name_valid_log,
                    )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.train_name_valid_log,
            )

        except Exception as e:
            self.log_writer.raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.train_name_valid_log,
            )

    def validate_col_length(self, NumberofColumns):
        """
        Method Name :   validate_col_length
        Description :   This method is used for validating the column length of the csv file

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.validate_col_length.__name__

        try:
            self.log_writer.start_log(
                key="start",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.train_col_valid_log,
            )

            csv_file_objs = self.s3_obj.get_file_objects_from_s3(
                bucket=self.train_data_bucket,
                filename=self.good_train_data_dir,
                db_name=self.db_name,
                collection_name=self.train_col_valid_log,
            )

            for f in csv_file_objs:
                file = f.key

                abs_f = file.split("/")[-1]

                if file.endswith(".csv"):
                    csv = convert_object_to_dataframe(
                        f,
                        db_name=self.db_name,
                        collection_name=self.train_col_valid_log,
                    )

                    if csv.shape[1] == NumberofColumns:
                        pass

                    else:
                        dest_f = self.bad_train_data_dir + "/" + abs_f

                        self.s3_obj.move_data_to_other_bucket(
                            src_bucket=self.train_data_bucket,
                            src_file=file,
                            dest_bucket=self.train_data_bucket,
                            dest_file=dest_f,
                            db_name=self.db_name,
                            collection_name=self.train_col_valid_log,
                        )

                else:
                    pass

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.train_col_valid_log,
            )

        except Exception as e:
            self.log_writer.raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.train_col_valid_log,
            )

    def validate_missing_values_in_col(self):
        """
        Method Name :   validate_missing_values_in_col
        Description :   This method is used for validating the missing values in columns

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.validate_missing_values_in_col.__name__

        try:
            self.log_writer.start_log(
                key="start",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.train_missing_value_log,
            )

            csv_file_objs = self.s3_obj.get_file_objects_from_s3(
                bucket=self.train_data_bucket,
                filename=self.good_train_data_dir,
                db_name=self.db_name,
                collection_name=self.train_missing_value_log,
            )

            for f in csv_file_objs:
                file = f.key

                abs_f = file.split("/")[-1]

                if abs_f.endswith(".csv"):
                    csv = convert_object_to_dataframe(
                        f,
                        db_name=self.db_name,
                        collection_name=self.train_missing_value_log,
                    )

                    count = 0

                    for cols in csv:
                        if (len(csv[cols]) - csv[cols].count()) == len(csv[cols]):
                            count += 1

                            dest_f = self.bad_train_data_dir + "/" + abs_f

                            self.s3_obj.move_data_to_other_bucket(
                                src_bucket=self.train_data_bucket,
                                src_file=file,
                                dest_bucket=self.train_data_bucket,
                                dest_file=dest_f,
                                db_name=self.db_name,
                                collection_name=self.train_missing_value_log,
                            )

                            break

                    if count == 0:
                        dest_f = self.good_train_data_dir + "/" + abs_f

                        self.s3_obj.upload_df_as_csv_to_s3(
                            data_frame=csv,
                            file_name=abs_f,
                            bucket=self.train_data_bucket,
                            dest_file_name=dest_f,
                            db_name=self.db_name,
                            collection_name=self.train_missing_value_log,
                        )

                else:
                    pass

                self.log_writer.start_log(
                    key="exit",
                    class_name=self.class_name,
                    method_name=method_name,
                    db_name=self.db_name,
                    collection_name=self.train_missing_value_log,
                )

        except Exception as e:
            self.log_writer.raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.train_missing_value_log,
            )
