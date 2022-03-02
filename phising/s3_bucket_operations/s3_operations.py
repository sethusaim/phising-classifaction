import json
import os
import pickle
from io import StringIO

import boto3
import pandas as pd
from botocore.exceptions import ClientError
from utils.logger import App_Logger
from utils.model_utils import Model_Utils
from utils.read_params import read_params


class S3_Operation:
    """
    Description :   This method is used for all the S3 bucket_name operations

    Version     :   1.2
    Revisions   :   moved to setup to cloud
    """

    def __init__(self):
        self.log_writer = App_Logger()

        self.config = read_params()

        self.class_name = self.__class__.__name__

        self.model_utils = Model_Utils()

        self.file_format = self.config["model_utils"]["save_format"]

    def get_s3_client(self, table_name):
        """
        Method Name :   get_s3_client
        Description :   This method is used getting the s3 client 

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_s3_client.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            s3_client = boto3.client("s3")

            self.log_writer.log(table_name=table_name, log_message="Got s3 client")

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

            return s3_client

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def get_s3_resource(self, table_name):
        """
        Method Name :   get_s3_resource
        Description :   This method is used getting the s3 resource

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_s3_resource.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            s3_resource = boto3.resource("s3")

            self.log_writer.log(table_name=table_name, log_message="Got s3 resource")

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

            return s3_resource

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def read_object(self, object, table_name, decode=True, make_readable=False):
        """
        Method Name :   read_object
        Description :   This method is used reading the s3 object

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.read_object.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            func = (
                lambda: object.get()["Body"].read().decode()
                if decode is True
                else object.get()["Body"].read()
            )

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Read the s3 object with decode as {decode}",
            )

            conv_func = lambda: StringIO(func()) if make_readable is True else func()

            self.log_writer.log(
                table_name=table_name,
                log_message=f"read the s3 object with make_readable as {make_readable}",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

            return conv_func()

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def read_text(self, file_name, bucket_name, table_name):
        """
        Method Name :   read_json
        Description :   This method is used for loading a json file from s3 bucket_name (schema file)

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.read_text.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            txt_obj = self.get_file_object(
                file_name=file_name, bucket_name=bucket_name, table_name=table_name
            )

            content = self.read_object(object=txt_obj, table_name=table_name)

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Read {file_name} file as text from {bucket_name} bucket_name",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

            return content

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def read_json(self, file_name, bucket_name, table_name):
        """
        Method Name :   read_json
        Description :   This method is used for loading a json file from s3 bucket_name (schema file)

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
            f_obj = self.get_file_object(
                file_name=file_name, bucket_name=bucket_name, table_name=table_name
            )

            json_content = self.read_object(object=f_obj, table_name=table_name)

            dic = json.loads(json_content)

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Read {file_name} from {bucket_name} bucket_name",
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

    def get_df_from_object(self, object, table_name):
        """
        Method Name :   get_df_from_object
        Description :   This method is used for converting object to dataframe

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_df_from_object.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            content = self.read_object(
                object=object, table_name=table_name, make_readable=True
            )

            df = pd.read_csv(content)

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

            return df

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def read_csv(self, file_name, bucket_name, table_name):
        method_name = self.read_csv.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            csv_obj = self.get_file_object(
                file_name=file_name, bucket_name=bucket_name, table_name=table_name,
            )

            df = self.get_df_from_object(object=csv_obj, table_name=table_name)

            self.log_writer.log(
                table_name=table_name,
                log_info=f"Read {file_name} csv file from {bucket_name} bucket",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

            return df

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def read_csv_from_folder(self, folder_name, bucket_name, table_name):
        method_name = self.read_csv_from_folder.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )
        try:
            files = self.get_files_from_folder(
                folder_name=folder_name, bucket_name=bucket_name, table_name=table_name,
            )

            lst = [
                (
                    self.read_csv(
                        file_name=f, bucket_name=bucket_name, table_name=table_name,
                    ),
                    f,
                    f.split("/")[-1],
                )
                for f in files
            ]

            self.log_writer.log(
                table_name=table_name,
                log_info=f"Read csv files from {folder_name} folder from {bucket_name} bucket",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

            return lst

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def load_object(self, object, bucket_name, table_name):
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
            s3_resource = self.get_s3_resource(table_name=table_name)

            s3_resource.Object(bucket_name, object).load()

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Loaded {object} from {bucket_name} bucket_name",
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

    def create_folder(self, folder_name, bucket_name, table_name):
        """
        Method Name :   create_folder
        Description :   This method is used for creating a folder in s3 bucket_name

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
            self.load_object(bucket_name=bucket_name, object=folder_name)

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

        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                self.log_writer.log(
                    table_name=table_name,
                    log_message=f"{folder_name} folder does not exist,creating new one",
                )

                self.put_object(
                    object=folder_name, bucket_name=bucket_name, table_name=table_name
                )

                self.log_writer.log(
                    table_name=table_name,
                    log_message=f"{folder_name} folder created in {bucket_name} bucket_name",
                )

            else:
                self.log_writer.log(
                    table_name=table_name,
                    log_message=f"Error occured in creating {folder_name} folder",
                )

                self.log_writer.exception_log(
                    error=e,
                    class_name=self.class_name,
                    method_name=method_name,
                    table_name=table_name,
                )

    def put_object(self, object, bucket_name, table_name):
        """
        Method Name :   put_object
        Description :   This method is used for putting any object in s3 bucket_name

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
            s3_client = self.get_s3_client(table_name=table_name)

            s3_client.put_object(Bucket=bucket_name, Key=(object + "/"))

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Created {object} folder in {bucket_name} bucket",
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

    def upload_file(self, from_file, to_file, bucket_name, table_name, remove=True):
        """
        Method Name :   upload_file
        Description :   This method is used for uploading the files to s3 bucket_name

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
                log_message=f"Uploading {from_file} to s3 bucket_name {bucket_name}",
            )

            s3_resource = self.get_s3_resource(table_name=table_name)

            s3_resource.meta.client.upload_file(from_file, bucket_name, to_file)

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Uploaded {from_file} to s3 bucket_name {bucket_name}",
            )

            if remove is True:
                self.log_writer.log(
                    table_name=table_name,
                    log_message=f"Option remove is set {remove}..deleting the file",
                )

                os.remove(from_file)

                self.log_writer.log(
                    table_name=table_name,
                    log_message=f"Removed the local copy of {from_file}",
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

    def get_bucket(self, bucket_name, table_name):
        """
        Method Name :   get_bucket
        Description :   This method is used for getting the bucket_name from s3

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
            s3_resource = self.get_s3_resource(table_name=table_name)

            bucket_name = s3_resource.Bucket(bucket_name)

            self.log_writer.log(
                table_name=table_name, log_message=f"Got {bucket_name} bucket_name",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

            return bucket_name

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def copy_data(self, from_file, from_bucket, to_file, to_bucket, table_name):
        """
        Method Name :   copy_data
        Description :   This method is used for copying the data from one bucket_name to another

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
            copy_source = {"Bucket": from_bucket, "Key": from_file}

            s3_resource = self.get_s3_resource(table_name=table_name)

            s3_resource.meta.client.copy(copy_source, to_bucket, to_file)

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Copied data from bucket_name {from_bucket} to bucket_name {to_bucket}",
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

    def delete_file(self, file_name, bucket_name, table_name):
        """
        Method Name :   delete_file
        Description :   This method is used for deleting any file from s3 bucket_name

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
            s3_resource = self.get_s3_resource(table_name=table_name)

            s3_resource.Object(bucket_name, file_name).delete()

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Deleted {file_name} from bucket_name {bucket_name}",
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

    def move_data(self, from_file, from_bucket, to_bucket, to_file, table_name):
        """
        Method Name :   move_data
        Description :   This method is used for moving the data from one bucket_name to another bucket_name

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
                from_bucket=from_bucket,
                from_file=from_file,
                to_bucket=to_bucket,
                to_file=to_file,
                table_name=table_name,
            )

            self.delete_file(
                bucket_name=from_bucket, file=from_file, table_name=table_name,
            )

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Moved {from_file} from bucket_name {from_bucket} to {to_bucket}",
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

    def get_files_from_folder(self, folder_name, bucket_name, table_name):
        """
        Method Name :   get_files_from_folder
        Description :   This method is used for getting the file names from s3 bucket_name

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_files_from_folder.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            lst = self.get_file_object(
                bucket_name=bucket_name, table_name=table_name, file_name=folder_name,
            )

            list_of_files = [object.key for object in lst]

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Got list of files from bucket_name {bucket_name}",
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

    def get_file_object(self, file_name, bucket_name, table_name):
        """
        Method Name :   get_file_object
        Description :   This method is used for getting file contents from s3 bucket_name

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
            bucket = self.get_bucket(bucket_name=bucket_name, table_name=table_name,)

            lst_objs = [object for object in bucket.objects.filter(Prefix=file_name)]

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Got {file_name} from bucket_name {bucket_name}",
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

    def load_model(self, model_name, bucket_name, table_name):
        """
        Method Name :   load_model
        Description :   This method is used for loading the model from s3 bucket_name

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
            f_obj = self.get_file_object(
                file_name=model_name, bucket_name=bucket_name, table_name=table_name
            )

            model_obj = self.read_object(
                object=f_obj, table_name=table_name, decode=False
            )

            model = pickle.loads(model_obj)

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Loaded {model_name} from bucket_name {bucket_name}",
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

    def save_model(self, model, idx, model_dir, model_bucket, table_name):
        """
        Method Name :   save_model
        Description :   This method is used for saving a model to s3 bucket_name

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
            model_name = self.model_utils.get_model_name(
                model=model, table_name=table_name
            )

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
                log_message=f"Saved {model_name} model as {model_file} name",
            )

            bucket_model_path = model_dir + "/" + model_file

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Uploading {model_file} to {model_bucket} bucket",
            )

            self.upload_file(
                from_file=model_file,
                to_file=bucket_model_path,
                bucket_name=model_bucket,
                table_name=table_name,
            )

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Uploaded  {model_file} to {model_bucket} bucket_name",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

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
        self, data_frame, local_file_name, bucket_file_name, bucket_name, table_name
    ):
        """
        Method Name :   upload_df_as_csv
        Description :   This method is used for uploading a dataframe to s3 bucket_name as csv file

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
            data_frame.to_csv(local_file_name, index=None, header=True)

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Created a local copy of dataframe with name {local_file_name}",
            )

            self.upload_file(
                from_file=local_file_name,
                to_file=bucket_file_name,
                bucket_name=bucket_name,
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
