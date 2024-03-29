import os
import re
import shutil
import sys
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from phising.data_access.phising_data import PhisingData
from phising.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from phising.entity.config_entity import DataValidationConfig
from phising.exception import PhisingException
from phising.logger import logging
from phising.utils.main_utils import read_text, read_yaml


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ):
        self.data_ingestion_artifact = data_ingestion_artifact

        self.data_validation_config = data_validation_config

        self.phising_data = PhisingData()

    def values_from_schema(self) -> Tuple[int, int, str, int]:
        """
        Method Name :   values_from_schema
        Description :   This method gets schema values from the schema_training.json file

        Output      :   Schema values are extracted from the schema_training.json file
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logging.info("Entered values_from_schema method of class")

        try:
            dic: Dict = read_yaml(
                self.data_validation_config.data_validation_training_schema_path
            )

            logging.info(
                f"Loaded the {self.data_validation_config.data_validation_training_schema_path} file"
            )

            LengthOfDateStampInFile: int = dic["LengthOfDateStampInFile"]

            LengthOfTimeStampInFile: int = dic["LengthOfTimeStampInFile"]

            column_names: str = dic["ColName"]

            NumberofColumns: int = dic["NumberofColumns"]

            message = (
                "LengthOfDateStampInFile:: %s" % LengthOfDateStampInFile
                + "\t"
                + "LengthOfTimeStampInFile:: %s" % LengthOfTimeStampInFile
                + "\t "
                + "NumberofColumns:: %s" % NumberofColumns
            )

            logging.info(f"Values from schema are : {message}")

            logging.info("Exited values_from_schema method of class")

            return (
                LengthOfDateStampInFile,
                LengthOfTimeStampInFile,
                column_names,
                NumberofColumns,
            )

        except Exception as e:
            raise PhisingException(e, sys)

    def validate_raw_fname(
        self, LengthOfDateStampInFile: int, LengthOfTimeStampInFile: int
    ) -> None:
        """
        Method Name :   validate_raw_fname
        Description :   This method validates the raw file name based on regex pattern and schema values

        Output      :   Raw file names are validated, good file names are stored in good data folder and rest is stored in bad data
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logging.info("Entered validate_raw_fname method of DataValidation class")

        try:
            onlyfiles: List[str] = os.listdir(
                self.data_ingestion_artifact.feature_store_folder_path
            )

            logging.info(
                f"Got a list of files from {self.data_ingestion_artifact.feature_store_folder_path}"
            )

            regex: str = read_text(
                self.data_validation_config.data_validation_regex_path
            )

            logging.info(
                f"Got regex pattern {regex} from {self.data_validation_config.data_validation_regex_path}"
            )

            for fname in onlyfiles:
                data_ingestion_fname: str = (
                    self.data_ingestion_artifact.feature_store_folder_path + "/" + fname
                )

                os.makedirs(
                    self.data_validation_config.data_validation_valid_data_dir,
                    exist_ok=True,
                )

                os.makedirs(
                    self.data_validation_config.data_validation_invalid_data_dir,
                    exist_ok=True,
                )

                if re.match(regex, fname):
                    splitAtDot = re.split(".csv", fname)

                    splitAtDot = re.split("_", splitAtDot[0])

                    if len(splitAtDot[1]) == LengthOfDateStampInFile:
                        if len(splitAtDot[2]) == LengthOfTimeStampInFile:
                            shutil.copy(
                                data_ingestion_fname,
                                self.data_validation_config.data_validation_valid_data_dir,
                            )

                            logging.info(
                                f"Copied {data_ingestion_fname} file to {self.data_validation_config.data_validation_valid_data_dir} folder"
                            )

                        else:
                            shutil.copy(
                                data_ingestion_fname,
                                self.data_validation_config.data_validation_invalid_data_dir,
                            )

                            logging.info(
                                f"Copied {data_ingestion_fname} file to {self.data_validation_config.data_validation_invalid_data_dir} folder"
                            )

                    else:
                        shutil.copy(
                            data_ingestion_fname,
                            self.data_validation_config.data_validation_invalid_data_dir,
                        )

                        logging.info(
                            f"Copied {data_ingestion_fname} file to {self.data_validation_config.data_validation_invalid_data_dir}"
                        )

                else:
                    shutil.copy(
                        data_ingestion_fname,
                        self.data_validation_config.data_validation_invalid_data_dir,
                    )

                    logging.info(
                        f"Copied {data_ingestion_fname} file to {self.data_validation_config.data_validation_invalid_data_dir}"
                    )

            logging.info("Exited validate_raw_fname method of DataValidation class")

        except Exception as e:
            raise PhisingException(e, sys)

    def validate_col_length(self, NumberofColumns: int) -> None:
        """
        Method Name :   validate_col_length
        Description :   This method validates the column length based on number of columns as mentioned in schema values

        Output      :   The files' columns length are validated and good data is stored in good data folder and rest is stored in bad data folder
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logging.info("Entered validate_col_length method of DataValidation class")

        try:
            lst: Tuple[pd.DataFrame, str, str] = self.phising_data.read_csv_from_folder(
                folder_name=self.data_validation_config.data_validation_valid_data_dir
            )

            logging.info(
                f"Got a list of tuple of dataframe,filename and absolute filename from {self.data_validation_config.data_validation_valid_data_dir} folder"
            )

            for _, f in enumerate(lst):
                df: pd.DataFrame = f[0]

                file: str = f[1]

                if df.shape[1] == NumberofColumns:
                    pass

                else:
                    shutil.move(
                        file,
                        self.data_validation_config.data_validation_invalid_data_dir,
                    )

                    logging.info(
                        f"Moved {file} file to {self.data_validation_config.data_validation_invalid_data_dir} folder"
                    )

            logging.info("Exited validate_col_length method of DataValidation class")

        except Exception as e:
            raise PhisingException(e, sys)

    def validate_missing_values_in_col(self) -> None:
        """
        Method Name :   validate_missing_values_in_col
        Description :   This method validates the missing values in columns

        Output      :   Missing columns are validated, and good data is stored in good data folder and rest is to stored in bad data folder
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logging.info(
            "Entered validate_missing_values_in_col method of DataValidation class"
        )

        try:
            lst: Tuple[pd.DataFrame, str, str] = self.phising_data.read_csv_from_folder(
                folder_name=self.data_validation_config.data_validation_valid_data_dir
            )

            logging.info(
                f"Got a list of tuple of dataframe,filename and absolute filename from {self.data_validation_config.data_validation_valid_data_dir} folder"
            )

            for _, f in enumerate(lst):
                df: pd.DataFrame = f[0]

                file: str = f[1]

                count: int = 0

                for cols in df:
                    if (len(df[cols]) - df[cols].count()) == len(df[cols]):
                        count += 1

                        shutil.move(
                            file,
                            self.data_validation_config.data_validation_invalid_data_dir,
                        )

                        logging.info(
                            f"Moved {file} file to {self.data_validation_config.data_validation_invalid_data_dir} folder"
                        )

                        break

            logging.info(
                "Exited validate_missing_values_in_col method of DataValidation class"
            )

        except Exception as e:
            raise PhisingException(e, sys)

    def check_validation_status(self) -> bool:
        logging.info("Entered check_validation_status method of DataValidation class")

        try:
            status: bool = False

            if (
                len(
                    os.listdir(
                        self.data_validation_config.data_validation_valid_data_dir
                    )
                )
                != 0
            ):
                status: bool = True

            logging.info(f"Validation status is to {status}")

            logging.info(
                "Exited check_validation_status method of DataValidation class"
            )

            return status

        except Exception as e:
            raise PhisingException(e, sys)

    @staticmethod
    def merge_batch_data(
        folder_name: str,
        input_file: str,
    ) -> List[pd.DataFrame]:
        logging.info("Entered merge_batch_data method of DataIngestion class")

        try:
            lst: List[pd.DataFrame] = [
                pd.read_csv(
                    folder_name + "/" + f,
                )
                for f in os.listdir(folder_name)
            ]

            new_df: pd.DataFrame = pd.concat(lst, ignore_index=True)

            new_df.to_csv(input_file, index=False, header=True)

            logging.info("Exited merge_batch_data method of DataIngestion class")

            return new_df

        except Exception as e:
            raise PhisingException(e, sys)

    def split_data_as_train_test(
        self, dataframe: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Method Name :   split_data_as_train_test
        Description :   This method splits the dataframe into train set and test set based on split ratio

        Output      :   Folder is created in s3 bucket
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logging.info("Entered split_data_as_train_test method of Data_Ingestion class")

        try:
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_validation_config.data_validation_split_ratio,
            )

            logging.info("Performed train test split on the dataframe")

            logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )

            return train_set, test_set

        except Exception as e:
            raise PhisingException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Method Name :   initiate_data_validation
        Description :   This method initiates the data validation

        Output      :   Data Validation is done and artifacts are stored in artifacts folder
        On Failure  :   Raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logging.info("Entered initiate_data_validation method of DataValidation class")

        try:
            (
                LengthOfDateStampInFile,
                LengthOfTimeStampInFile,
                _,
                noofcolumns,
            ) = self.values_from_schema()

            self.validate_raw_fname(
                LengthOfDateStampInFile=LengthOfDateStampInFile,
                LengthOfTimeStampInFile=LengthOfTimeStampInFile,
            )

            self.validate_col_length(NumberofColumns=noofcolumns)

            self.validate_missing_values_in_col()

            if self.check_validation_status() is True:
                data: List[pd.DataFrame] = self.merge_batch_data(
                    folder_name=self.data_validation_config.data_validation_valid_data_dir,
                    input_file=self.data_validation_config.merged_file_path,
                )

                train_df, test_df = self.split_data_as_train_test(dataframe=data)

                train_df.to_csv(
                    self.data_validation_config.training_file_path,
                    index=False,
                    header=True,
                )

                test_df.to_csv(
                    self.data_validation_config.testing_file_path,
                    index=False,
                    header=True,
                )

            else:
                raise Exception(
                    f"No valid data csv files are found. {self.data_validation_config.data_validation_valid_data_dir} is empty"
                )

            data_validation_artifact: DataValidationArtifact = DataValidationArtifact(
                valid_data_dir=self.data_validation_config.data_validation_valid_data_dir,
                invalid_data_dir=self.data_validation_config.data_validation_invalid_data_dir,
                training_file_path=self.data_validation_config.training_file_path,
                testing_file_path=self.data_validation_config.testing_file_path,
            )

            logging.info(f"Data Validation Artifact is : {data_validation_artifact}")

            logging.info(
                "Exited initiate_data_validation method of DataValidation class"
            )

            return data_validation_artifact

        except Exception as e:
            raise PhisingException(e, sys)
