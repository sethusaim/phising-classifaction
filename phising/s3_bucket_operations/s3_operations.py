import json
import os
import pickle
from io import StringIO

import boto3
import botocore
import pandas as pd
from utils.logger import app_logger
from utils.model_utils import get_model_name
from utils.read_params import read_params


class s3_operations:
    """
    Description :   This method is used for all the S3 bucket operations

    Version     :   1.2
    Revisions   :   moved to setup to cloud
    """

    def __init__(self):
        self.s3_client = boto3.client("s3")

        self.s3_resource = boto3.resource("s3")

        self.log_writer = app_logger()

        self.config = read_params()

        self.class_name = self.__class__.__name__

        self.file_format = self.config["model_utils"]["save_format"]

        self.train_data_bucket = self.config["s3_bucket"]["scania_train_data_bucket"]

        self.good_train_data_dir = self.config["data"]["train"]["good_data_dir"]

        self.bad_train_data_dir = self.config["data"]["train"]["bad_data_dir"]

        self.input_files_bucket = self.config["s3_bucket"]["input_files_bucket"]

        self.prod_model_dir = self.config["models_dir"]["prod"]

        self.stag_model_dir = self.config["models_dir"]["stag"]

        self.pred_file_name = self.config["export_pred_csv_file"]

        self.trained_model_dir = self.config["models_dir"]["trained"]

    def read_object(self, obj, table_name, decode=True):
        """
        Method Name :   read_object
        Description :   This method is used for reading a object from s3 bucket

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.read_object.__name__

        self.log_writer.start_log(
            key="start",
            class_name=__file__,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            func = (
                lambda: obj.get()["Body"].read().decode()
                if decode
                else obj.get()["Body"].read()
            )

            content = func()

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Read the object with decode as {decode}",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=__file__,
                method_name=method_name,
                table_name=table_name,
            )

            return content

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=__file__,
                method_name=method_name,
                table_name=table_name,
            )

    def read_csv(self, bucket, file_name, table_name, folder=False):
        """
        Method Name :   read_csv
        Description :   This method is used for reading the csv file from s3 bucket

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.read_csv.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            csv_obj = self.get_file_object(
                bucket=bucket, filename=file_name, table_name=table_name
            )

            if folder is True:
                lst = [
                    (
                        self.convert_object_to_dataframe(
                            obj=obj, table_name=table_name
                        ),
                        obj.key,
                        obj.key.split("/")[-1],
                    )
                    for obj in csv_obj
                ]

                return lst

            else:
                df = self.convert_object_to_dataframe(
                    obj=csv_obj, table_name=table_name
                )

                return df

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def read_json(self, obj, table_name):
        """
        Method Name :   read_json
        Description :   This method is used for converting the s3 object to json

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.read_json.__name__

        self.log_writer.start_log(
            key="start",
            class_name=__file__,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            res = self.convert_object_to_bytes(obj=obj, table_name=table_name)

            dic = json.loads(res)

            self.log_writer.log(
                table_name=table_name, log_message=f"Converted {obj} to json",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=__file__,
                method_name=method_name,
                table_name=table_name,
            )

            return dic

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=__file__,
                method_name=method_name,
                table_name=table_name,
            )

    def read_pickle(self, obj, table_name):
        """
        Method Name :   read_pickle
        Description :   This method is used for converting the s3 obj to pickle format

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.read_pickle.__name__

        self.log_writer.start_log(
            key="start",
            class_name=__file__,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            model_content = self.read_object(obj, decode=False, table_name=table_name)

            model = pickle.loads(model_content)

            self.log_writer.log(
                table_name=table_name, log_message=f"Loaded {obj} as pickle model",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=__file__,
                method_name=method_name,
                table_name=table_name,
            )

            return model

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=__file__,
                method_name=method_name,
                table_name=table_name,
            )

    def load_object(self, bucket_name, obj, table_name):
        """
        Method Name :   load_object
        Description :   This method is used for loading a object from s3 bucket

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.load_object.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            self.s3_resource.Object(bucket_name, obj).load()

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Loaded {obj} from {bucket_name} bucket",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def convert_object_to_dataframe(self, obj, table_name):
        """
        Method Name :   convert_object_to_dataframe
        Description :   This method is used for converting the s3 object to dataframe

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.convert_object_to_dataframe.__name__

        self.log_writer.start_log(
            key="start",
            class_name=__file__,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            content = self.convert_object_to_bytes(obj, table_name=table_name)

            data = self.make_readable(data=content, table_name=table_name)

            df = pd.read_csv(data)

            self.log_writer.log(
                table_name=table_name, log_message=f"Converted {obj} to dataframe",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=__file__,
                method_name=method_name,
                table_name=table_name,
            )

            return df

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=__file__,
                method_name=method_name,
                table_name=table_name,
            )

    def convert_object_to_pickle(self, obj, table_name):
        """
        Method Name :   convert_object_to_pickle
        Description :   This method is used for converting the s3 obj to pickle format

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.convert_object_to_pickle.__name__

        self.log_writer.start_log(
            key="start",
            class_name=__file__,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            model_content = self.read_object(obj, decode=False, table_name=table_name)

            model = pickle.loads(model_content)

            self.log_writer.log(
                table_name=table_name, log_message=f"Loaded {obj} as pickle model",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=__file__,
                method_name=method_name,
                table_name=table_name,
            )

            return model

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=__file__,
                method_name=method_name,
                table_name=table_name,
            )

    def convert_object_to_bytes(self, obj, table_name):
        """
        Method Name :   convert_object_to_bytes
        Description :   This method is used for converting the s3 object to bytes

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.convert_object_to_bytes.__name__

        self.log_writer.start_log(
            key="start",
            class_name=__file__,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            content = self.read_object(obj, decode=True, table_name=table_name,)

            self.log_writer.log(
                table_name=table_name, log_message=f"Converted {obj} to bytes",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=__file__,
                method_name=method_name,
                table_name=table_name,
            )

            return content

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=__file__,
                method_name=method_name,
                table_name=table_name,
            )

    def make_readable(self, data, table_name):
        """
        Method Name :   make_readable
        Description :   This method is used for converting the bytes object to string data

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.make_readable.__name__

        self.log_writer.start_log(
            key="start",
            class_name=__file__,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            f = StringIO(data)

            self.log_writer.log(
                table_name=table_name,
                log_message="Converted bytes content to string content using StringIO",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=__file__,
                method_name=method_name,
                table_name=table_name,
            )

            return f

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=__file__,
                method_name=method_name,
                table_name=table_name,
            )

    def convert_obj_to_json(self, obj, table_name):
        """
        Method Name :   convert_obj_to_json
        Description :   This method is used for converting the s3 object to json

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.convert_obj_to_json.__name__

        self.log_writer.start_log(
            key="start",
            class_name=__file__,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            res = self.convert_object_to_bytes(obj=obj, table_name=table_name)

            dic = json.loads(res)

            self.log_writer.log(
                table_name=table_name, log_message=f"Converted {obj} to json",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=__file__,
                method_name=method_name,
                table_name=table_name,
            )

            return dic

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=__file__,
                method_name=method_name,
                table_name=table_name,
            )

    def find_correct_model_file(self, cluster_number, bucket_name, table_name):
        """
        Method Name :   find_correct_model_file
        Description :   This method is used for finding the correct model file during prediction

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.find_correct_model_file.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            prod_model_dir = self.config["models_dir"]["prod"]

            list_of_files = self.get_files(
                bucket=bucket_name, folder_name=prod_model_dir, table_name=table_name,
            )

            for file in list_of_files:
                try:
                    if file.index(str(cluster_number)) != -1:
                        model_name = file

                except:
                    continue

            model_name = model_name.split(".")[0]

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Got {model_name} from {prod_model_dir} folder in {bucket_name} bucket",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

            return model_name

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def delete_pred_file(self, table_name):
        """
        Method Name :   delete_pred_file
        Description :   This method is used for deleting the existing prediction batch file

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.delete_pred_file.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            self.s3_resource.Object(self.input_files_bucket, self.pred_file_name).load()

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Found existing prediction batch file. Deleting it.",
            )

            self.delete_file(
                bucket_name=self.input_files_bucket,
                file=self.pred_file_name,
                table_name=table_name,
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                pass

            else:
                self.log_writer.exception_log(
                    error=e,
                    class_name=self.class_name,
                    method_name=method_name,
                    table_name=table_name,
                )

    def create_folder(self, bucket_name, folder_name, table_name):
        """
        Method Name :   create_folder
        Description :   This method is used for creating a folder in s3 bucket

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.create_folder.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            self.s3_resource.Object(bucket_name, folder_name).load()

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Folder {folder_name} already exists.",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                self.put_object(
                    bucket=bucket_name, folder_name=folder_name, table_name=table_name,
                )

            else:
                self.log_writer.log(
                    table_name=table_name,
                    log_message="Error occured in creating folder",
                )

                self.log_writer.exception_log(
                    error=e,
                    class_name=self.class_name,
                    method_name=method_name,
                    table_name=table_name,
                )

    def put_object(self, bucket, folder_name, table_name):
        """
        Method Name :   put_object
        Description :   This method is used for putting any object in s3 bucket

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.put_object.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            self.s3_client.put_object(Bucket=bucket, Key=(folder_name + "/"))

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Created {folder_name} folder in {bucket} bucket",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def upload_file(self, src_file, table_name, bucket, dest_file, remove=True):
        """
        Method Name :   upload_file
        Description :   This method is used for uploading the files to s3 bucket

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.upload_file.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            self.log_writer.log(
                table_name=table_name,
                log_message=f"Uploading {src_file} to s3 bucket {bucket}",
            )

            self.s3_resource.meta.client.upload_file(src_file, bucket, dest_file)

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Uploaded {src_file} to s3 bucket {bucket}",
            )

            if remove:
                self.log_writer.log(
                    table_name=table_name,
                    log_message=f"Option remove is set {remove}..deleting the file",
                )

                os.remove(src_file)

                self.log_writer.log(
                    table_name=table_name,
                    log_message=f"Removed the local copy of {src_file}",
                )

                self.log_writer.start_log(
                    key="exit",
                    class_name=self.class_name,
                    method_name=method_name,
                    table_name=table_name,
                )

            else:
                self.log_writer.log(
                    table_name=table_name,
                    log_message=f"Option remove is set {remove}, not deleting the file",
                )

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def get_bucket(self, bucket, table_name):
        """
        Method Name :   get_bucket
        Description :   This method is used for getting the bucket from s3

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_bucket.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            bucket = self.s3_resource.Bucket(bucket)

            self.log_writer.log(
                table_name=table_name, log_message=f"Got {bucket} s3 bucket",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

            return bucket

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def copy_data(self, src_bucket, src_file, dest_bucket, dest_file, table_name):
        """
        Method Name :   copy_data
        Description :   This method is used for copying the data from one bucket to another

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.copy_data.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            copy_source = {"Bucket": src_bucket, "Key": src_file}

            self.s3_resource.meta.client.copy(copy_source, dest_bucket, dest_file)

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Copied data from bucket {src_bucket} to bucket {dest_bucket}",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def delete_file(self, bucket, file, table_name):
        """
        Method Name :   delete_file
        Description :   This method is used for deleting any file from s3 bucket

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.delete_file.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            self.s3_resource.Object(bucket, file).delete()

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Deleted {file} from bucket {bucket}",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def move_data(self, src_bucket, src_file, dest_bucket, dest_file, table_name):
        """
        Method Name :   move_data
        Description :   This method is used for moving the data from one bucket to another bucket

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.move_data.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            self.copy_data(
                src_bucket=src_bucket,
                src_file=src_file,
                dest_bucket=dest_bucket,
                dest_file=dest_file,
                table_name=table_name,
            )

            self.delete_file(
                bucket=src_bucket, file=src_file, table_name=table_name,
            )

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Moved {src_file} from bucket {src_bucket} to {dest_bucket}",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def get_files(self, bucket, folder_name, table_name):
        """
        Method Name :   get_files
        Description :   This method is used for getting the file names from s3 bucket

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_files.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            lst = self.get_file_object(
                bucket=bucket, table_name=table_name, filename=folder_name,
            )

            list_of_files = [obj.key for obj in lst]

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Got list of files from bucket {bucket}",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

            return list_of_files

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def get_file_object(self, bucket, filename, table_name):
        """
        Method Name :   get_file_object
        Description :   This method is used for getting file contents from s3 bucket

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_file_object.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            s3_bucket = self.get_bucket(bucket=bucket, table_name=table_name,)

            lst_objs = [obj for obj in s3_bucket.objects.filter(Prefix=filename)]

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Got {filename} from bucket {bucket}",
            )

            func = lambda x: x[0] if len(x) == 1 else x

            file_objs = func(lst_objs)

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

            return file_objs

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def load_model(self, bucket, model_name, table_name):
        """
        Method Name :   load_model
        Description :   This method is used for loading the model from s3 bucket

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.load_model.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            model_obj = self.get_file_object(
                bucket=bucket, filename=model_name, table_name=table_name,
            )

            model = self.utils.read_pickle(obj=model_obj, table_name=table_name)

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Loaded {model_name} from bucket {bucket}",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

            return model

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def read_json(self, bucket, filename, table_name):
        """
        Method Name :   read_json
        Description :   This method is used for loading a json file from s3 bucket (schema file)

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.read_json.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            res = self.get_file_object(
                bucket=bucket, filename=filename, table_name=table_name,
            )

            dic = self.utils.read_json(obj=res, table_name=table_name)

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Got {filename} schema from bucket {bucket}",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

            return dic

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def create_dirs_for_good_bad_data(self, table_name):
        """
        Method Name :   create_dirs_for_good_bad_data
        Description :   This method is used for creating directory for good and bad data in s3 bucket

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.create_dirs_for_good_bad_data.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            self.create_folder(
                bucket_name=self.train_data_bucket,
                folder_name=self.good_train_data_dir,
                table_name=table_name,
            )

            self.create_folder(
                bucket_name=self.train_data_bucket,
                folder_name=self.bad_train_data_dir,
                table_name=table_name,
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def create_folders_for_prod_and_stag(self, bucket_name, table_name):
        """
        Method Name :   create_folders_for_prod_and_stag
        Description :   This method is used for creating production and staging folder in s3 bucket

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.create_folders_for_prod_and_stag.__name__

        self.log_writer.start_log(
            key="exit",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            self.create_folder(
                bucket_name=bucket_name,
                folder_name=self.prod_model_dir,
                table_name=table_name,
            )

            self.create_folder(
                bucket_name=bucket_name,
                folder_name=self.stag_model_dir,
                table_name=table_name,
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def save_model(self, idx, model, model_bucket, table_name):
        """
        Method Name :   save_model
        Description :   This method is used for saving a model to s3 bucket

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.save_model.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            model_name = get_model_name(model=model, table_name=table_name)

            func = (
                lambda: model_name + self.file_format
                if model_name == "KMeans"
                else model_name + str(idx) + self.file_format
            )

            model_file = func()

            with open(file=model_file, mode="wb") as f:
                pickle.dump(model, f)

            self.log_writer.log(
                table_name=table_name,
                log_message="Model File " + model_name + " saved. ",
            )

            s3_model_path = self.trained_model_dir + "/" + model_file

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Uploading {model_file} to {model_bucket} bucket",
            )

            self.upload_file(
                src_file=model_file,
                bucket=model_bucket,
                dest_file=s3_model_path,
                table_name=table_name,
            )

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Uploaded  {model_file} to {model_bucket} bucket",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

            return "success"

        except Exception as e:
            self.log_writer.log(
                table_name=table_name,
                log_message=f"Model file {model_name} could not be saved",
            )

            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def upload_df_as_csv(
        self, data_frame, file_name, bucket, dest_file_name, table_name
    ):
        """
        Method Name :   upload_df_as_csv
        Description :   This method is used for uploading a dataframe to s3 bucket as csv file

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.upload_df_as_csv.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            data_frame.to_csv(file_name, index=None, header=True)

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Created a local copy of dataframe with name {file_name}",
            )

            self.upload_file(
                src_file=file_name,
                bucket=bucket,
                dest_file=dest_file_name,
                table_name=table_name,
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )
