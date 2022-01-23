import numpy as np
import pandas as pd
from phising.s3_bucket_operations.s3_operations import S3_Operations
from utils.logger import App_Logger
from utils.read_params import read_params


class Preprocessor:
    """
    Description :   This class shall  be used to clean and transform the data before training.
    Version     :   1.2
    Revisions   :   moved setup to cloud
    """

    def __init__(self, db_name, collection_name):
        self.log_writer = App_Logger()

        self.config = read_params()

        self.class_name = self.__class__.__name__

        self.db_name = db_name

        self.collection_name = collection_name

        self.null_values_file = self.config["null_values_csv_file"]

        self.n_components = self.config["pca_model"]["n_components"]

        self.input_files_bucket = self.config["s3_bucket"]["input_files_bucket"]

        self.s3_obj = S3_Operations()

    def remove_columns(self, data, columns):
        """
        Method Name :   remove_columns
        Description :   This method removes the given columns from a pandas dataframe.
        Output      :   A pandas DataFrame after removing the specified columns.
        On Failure  :   Raise Exception
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.remove_columns.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            db_name=self.db_name,
            collection_name=self.collection_name,
        )

        self.data = data

        self.columns = columns
        try:
            self.useful_data = self.data.drop(labels=self.columns, axis=1)

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Dropped {columns} from {data}",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            return self.useful_data

        except Exception as e:
            self.log_writer.raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

    def separate_label_feature(self, data, label_column_name):
        """
        Method Name :   separate_label_feature
        Description :   This method separates the features and a Label Coulmns.
        Output      :   Returns two separate Dataframes, one containing features and the other containing Labels .
        On Failure  :   Raise Exception
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.separate_label_feature.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            db_name=self.db_name,
            collection_name=self.collection_name,
        )

        try:
            self.X = data.drop(labels=label_column_name, axis=1)

            self.Y = data[label_column_name]

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Separated {label_column_name} from {data}",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            return self.X, self.Y

        except Exception as e:
            self.log_writer.raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

    def drop_unnecessary_cols(self, data, cols_list):
        """
        Method Name :   drop_unnecessary_cols
        Description :   This method replaces drops unnecessary cols from the given data
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        try:
            data = data.drop(cols_list, axis=1)

            return data

        except Exception as e:
            raise e

    def replace_invalid_values(self, data):
        """
        Method Name :   replace_invalid_values
        Description :   This method replaces invalid values i.e. 'na' with np.nan
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.replace_invalid_values.__name__

        try:
            self.log_writer.start_log(
                key="start",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            for col in data.columns:
                count = data[col][data[col] == "'?'"].count()

                if count != 0:
                    data[col] = data[col].replace("'?'", np.nan)

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            return data

        except Exception as e:
            self.log_writer.raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

    def is_null_present(self, data):
        """
        Method Name :   is_null_present
        Description :   This method checks whether there are null values present in the pandas Dataframe or not.
        Output      :   Returns True if null values are present in the DataFrame, False if they are not present and
                        returns the list of columns for which null values are present.
        On Failure  :   Raise Exception
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.is_null_present.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            db_name=self.db_name,
            collection_name=self.collection_name,
        )

        null_present = False

        cols_with_missing_values = []

        cols = data.columns

        try:
            self.null_counts = data.isna().sum()

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Null values count is : {self.null_counts}",
            )

            for i in range(len(self.null_counts)):
                if self.null_counts[i] > 0:
                    null_present = True

                    cols_with_missing_values.append(cols[i])

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="created cols with missing values",
            )

            if null_present:
                self.log_writer.log(
                    db_name=self.db_name,
                    collection_name=self.collection_name,
                    log_message="null values were found the columns...preparing dataframe with null values",
                )

                self.dataframe_with_null = pd.DataFrame()

                self.dataframe_with_null["columns"] = data.columns

                self.dataframe_with_null["missing values count"] = np.asarray(
                    data.isna().sum()
                )

                self.log_writer.log(
                    db_name=self.db_name,
                    collection_name=self.collection_name,
                    log_message="Created dataframe with null values",
                )

                self.s3_obj.upload_df_as_csv_to_s3(
                    data_frame=self.dataframe_with_null,
                    file_name=self.null_values_file,
                    bucket=self.input_files_bucket,
                    dest_file_name=self.null_values_file,
                    db_name=self.db_name,
                    collection_name=self.collection_name,
                )

            else:
                self.log_writer.log(
                    db_name=self.db_name,
                    collection_name=self.collection_name,
                    log_message="No null values are present in cols. Skipped the creation of dataframe",
                )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            return null_present

        except Exception as e:
            self.log_writer.raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

    def encode_target_cols(self, data):
        """
        Method Name :   encode_target_cols
        Description :   This method encodes all the categorical values in the training set.
        Output      :   A Dataframe which has target values encoded.
        On Failure  :   Raise Exception
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.encode_target_cols.__name__

        try:
            self.log_writer.start_log(
                key="start",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            data["class"] = data["class"].map({"'p'": 1, "'e'": 2})

            for col in data.drop(["class"], axis=1).columns:
                data = pd.get_dummies(data, columns=[col])

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Encoded target cols in dataframe",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            return data

        except Exception as e:
            self.log_writer.raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

    def impute_missing_values(self, data):
        """
        Method Name : i mpute_missing_values
        Description :   This method replaces all the missing values in the Dataframe using mean values of the column.
        Output      :   A Dataframe which has all the missing values imputed.
        On Failure  :   Raise Exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.impute_missing_values.__name__

        try:
            self.log_writer.start_log(
                key="start",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            data = data[data.columns[data.isnull().mean() < 0.6]]

            data = data.apply(pd.to_numeric)

            for col in data.columns:
                data[col] = data[col].replace(np.nan, data[col].mean())

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            return data

        except Exception as e:
            self.log_writer.raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

    def get_columns_with_zero_std_deviation(self, data):
        """
        Method Name :   get_columns_with_zero_std_deviation
        Description :   This method finds out the columns which have a standard deviation of zero.
        Output      :   List of the columns with standard deviation of zero
        On Failure  :   Raise Exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_columns_with_zero_std_deviation.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            db_name=self.db_name,
            collection_name=self.collection_name,
        )

        try:
            data_n = data.describe()

            cols_to_drop = [x for x in data.columns if data_n[x]["std"] == 0]

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Got cols with zero standard deviation",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            return cols_to_drop

        except Exception as e:
            self.log_writer.raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )
