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
    Description :   This method is used for all the S3 bucket operations
    Written by  :   iNeuron Intelligence
    
    Version     :   1.2
    Revisions   :   Moved to setup to cloud 
    """

    def __init__(self):
        self.log_writer = App_Logger()

        self.config = read_params()

        self.class_name = self.__class__.__name__

        self.model_utils = Model_Utils()

        self.file_format = self.config["model_utils"]["save_format"]

        self.s3_client = boto3.client("s3")

        self.s3_resource = boto3.resource("s3")

    def read_object(self, object, log_file, decode=True, make_readable=False):
        """
        Method Name :   read_object
        Description :   This method reads the object with kwargs

        Output      :   A object is read with kwargs
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.read_object.__name__

        self.log_writer.start_log(
            "start", self.class_name, method_name,
        )

        try:
            func = (
                lambda: object.get()["Body"].read().decode()
                if decode is True
                else object.get()["Body"].read()
            )

            self.log_writer.log(
                log_file, f"Read the s3 object with decode as {decode}",
            )

            conv_func = lambda: StringIO(func()) if make_readable is True else func()

            self.log_writer.log(
                log_file, f"read the s3 object with make_readable as {make_readable}",
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name,
            )

            return conv_func()

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name,
            )

    def read_text(self, fname, bucket, log_file):
        """
        Method Name :   read_text
        Description :   This method reads the text data from s3 bucket

        Output      :   Text data is read from s3 bucket
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.read_text.__name__

        self.log_writer.start_log(
            "start", self.class_name, method_name,
        )

        try:
            txt_obj = self.get_file_object(fname, bucket, log_file)

            content = self.read_object(txt_obj, log_file)

            self.log_writer.log(
                log_file, f"Read {fname} file as text from {bucket} bucket",
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name,
            )

            return content

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name,
            )

    def read_json(self, fname, bucket, log_file):
        """
        Method Name :   read_json
        Description :   This method reads the json data from s3 bucket

        Output      :   Json data is read from s3 bucket
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.read_json.__name__

        self.log_writer.start_log(
            "start", self.class_name, method_name,
        )

        try:
            f_obj = self.get_file_object(fname, bucket, log_file)

            json_content = self.read_object(f_obj, log_file)

            dic = json.loads(json_content)

            self.log_writer.log(
                log_file, f"Read {fname} from {bucket} bucket",
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name,
            )

            return dic

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name,
            )

    def get_df_from_object(self, object, log_file):
        """
        Method Name :   get_df_from_object
        Description :   This method gets dataframe from object 

        Output      :   Dataframe is read from the object
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_df_from_object.__name__

        self.log_writer.start_log(
            "start", self.class_name, method_name,
        )

        try:
            content = self.read_object(object, make_readable=True)

            df = pd.read_csv(content)

            self.log_writer.start_log(
                "exit", self.class_name, method_name,
            )

            return df

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name,
            )

    def read_csv(self, fname, bucket, log_file):
        """
        Method Name :   read_csv
        Description :   This method reads the csv data from s3 bucket

        Output      :   A pandas series object consisting of runs for the particular experiment id
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.read_csv.__name__

        self.log_writer.start_log(
            "start", self.class_name, method_name,
        )

        try:
            csv_obj = self.get_file_object(fname, bucket,)

            df = self.get_df_from_object(csv_obj, log_file)

            self.log_writer.log(
                log_file, f"Read {fname} csv file from {bucket} bucket",
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name,
            )

            return df

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name,
            )

    def read_csv_from_folder(self, folder_name, bucket, log_file):
        """
        Method Name :   read_csv_from_folder
        Description :   This method reads the csv files from folder

        Output      :   A list of tuple of dataframe, along with absolute file name and file name is returned
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.read_csv_from_folder.__name__

        self.log_writer.start_log(
            "start", self.class_name, method_name,
        )
        try:
            files = self.get_files_from_folder(folder_name, bucket,)

            lst = [(self.read_csv(f, bucket,), f, f.split("/")[-1],) for f in files]

            self.log_writer.log(
                log_file,
                f"Read csv files from {folder_name} folder from {bucket} bucket",
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name,
            )

            return lst

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name,
            )

    def load_object(self, object, bucket, log_file):
        """
        Method Name :   load_object
        Description :   This method loads the object from s3 bucket

        Output      :   An object is loaded from s3 bucket
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.load_object.__name__

        self.log_writer.start_log(
            "start", self.class_name, method_name,
        )

        try:
            self.s3_resource.Object(bucket, object).load()

            self.log_writer.log(
                log_file, f"Loaded {object} from {bucket} bucket",
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name,
            )

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name,
            )

    def create_folder(self, folder_name, bucket, log_file):
        """
        Method Name :   create_folder
        Description :   This method creates a folder in s3 bucket

        Output      :   A folder is created in s3 bucket 
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.create_folder.__name__

        self.log_writer.start_log(
            "start", self.class_name, method_name,
        )

        try:
            self.load_object(bucket, folder_name)

            self.log_writer.log(
                log_file, f"Folder {folder_name} already exists.",
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name,
            )

        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                self.log_writer.log(
                    log_file, f"{folder_name} folder does not exist,creating new one",
                )

                self.put_object(folder_name, bucket, log_file)

                self.log_writer.log(
                    log_file, f"{folder_name} folder created in {bucket} bucket",
                )

            else:
                self.log_writer.log(
                    log_file, f"Error occured in creating {folder_name} folder",
                )

                self.log_writer.exception_log(
                    e, self.class_name, method_name,
                )

    def put_object(self, object, bucket, log_file):
        """
        Method Name :   put_object
        Description :   This method puts an object in s3 bucket

        Output      :   An object is put in s3 bucket
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.put_object.__name__

        self.log_writer.start_log(
            "start", self.class_name, method_name,
        )

        try:
            self.s3_client.put_object(Bucket=bucket, Key=(object + "/"))

            self.log_writer.log(
                log_file, f"Created {object} folder in {bucket} bucket",
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name,
            )

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name,
            )

    def upload_file(self, from_fname, to_file_name, bucket, log_file, remove=True):
        """
        Method Name :   upload_file
        Description :   This method uploades a file to s3 bucket with kwargs

        Output      :   A file is uploaded to s3 bucket
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.upload_file.__name__

        self.log_writer.start_log(
            "start", self.class_name, method_name,
        )

        try:
            self.log_writer.log(
                log_file, f"Uploading {from_fname} to s3 bucket {bucket}",
            )

            self.s3_resource.meta.client.upload_file(from_fname, bucket, to_file_name)

            self.log_writer.log(
                log_file, f"Uploaded {from_fname} to s3 bucket {bucket}",
            )

            if remove is True:
                self.log_writer.log(
                    log_file, f"Option remove is set {remove}..deleting the file",
                )

                os.remove(from_fname)

                self.log_writer.log(
                    log_file, f"Removed the local copy of {from_fname}",
                )

                self.log_writer.start_log(
                    "exit", self.class_name, method_name,
                )

            else:
                self.log_writer.log(
                    log_file, f"Option remove is set {remove}, not deleting the file",
                )

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name,
            )

    def get_bucket(self, bucket, log_file):
        """
        Method Name :   get_bucket
        Description :   This method gets the bucket from s3 

        Output      :   A s3 bucket name is returned based on the bucket
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_bucket.__name__

        self.log_writer.start_log(
            "start", self.class_name, method_name,
        )

        try:
            bucket = self.s3_resource.Bucket(bucket)

            self.log_writer.log(
                log_file, f"Got {bucket} bucket",
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name,
            )

            return bucket

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name,
            )

    def copy_data(self, from_fname, from_bucket, to_file_name, to_bucket, log_file):
        """
        Method Name :   copy_data
        Description :   This method copies the data from one bucket to another bucket

        Output      :   The data is copied from one bucket to another
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.copy_data.__name__

        self.log_writer.start_log(
            "start", self.class_name, method_name,
        )

        try:
            copy_source = {"Bucket": from_bucket, "Key": from_fname}

            self.s3_resource.meta.client.copy(copy_source, to_bucket, to_file_name)

            self.log_writer.log(
                log_file,
                f"Copied data from bucket {from_bucket} to bucket {to_bucket}",
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name,
            )

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name,
            )

    def delete_file(self, fname, bucket, log_file):
        """
        Method Name :   delete_file
        Description :   This method delete the file from s3 bucket

        Output      :   The file is deleted from s3 bucket
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.delete_file.__name__

        self.log_writer.start_log(
            "start", self.class_name, method_name,
        )

        try:
            self.s3_resource.Object(bucket, fname).delete()

            self.log_writer.log(
                log_file, f"Deleted {fname} from bucket {bucket}",
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name,
            )

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name,
            )

    def move_data(self, from_fname, from_bucket, to_file_name, to_bucket, log_file):
        """
        Method Name :   move_data
        Description :   This method moves the data from one bucket to other bucket

        Output      :   The data is moved from one bucket to another
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.move_data.__name__

        self.log_writer.start_log(
            "start", self.class_name, method_name,
        )

        try:
            self.copy_data(
                from_bucket, from_fname, to_bucket, to_file_name,
            )

            self.delete_file(
                from_bucket, file=from_fname,
            )

            self.log_writer.log(
                log_file,
                f"Moved {from_fname} from bucket {from_bucket} to {to_bucket}",
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name,
            )

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name,
            )

    def get_files_from_folder(self, folder_name, bucket, log_file):
        """
        Method Name :   get_files_from_folder
        Description :   This method gets the files a folder in s3 bucket

        Output      :   A list of files is returned
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_files_from_folder.__name__

        self.log_writer.start_log(
            "start", self.class_name, method_name,
        )

        try:
            lst = self.get_file_object(folder_name, bucket, log_file)

            list_of_files = [object.key for object in lst]

            self.log_writer.log(
                log_file, f"Got list of files from bucket {bucket}",
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name,
            )

            return list_of_files

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name,
            )

    def get_file_object(self, fname, bucket, log_file):
        """
        Method Name :   get_file_object
        Description :   This method gets the file object from s3 bucket

        Output      :   A file object is returned
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_file_object.__name__

        self.log_writer.start_log(
            "start", self.class_name, method_name,
        )

        try:
            bucket = self.get_bucket(bucket,)

            lst_objs = [object for object in bucket.objects.filter(Prefix=fname)]

            self.log_writer.log(
                log_file, f"Got {fname} from bucket {bucket}",
            )

            func = lambda x: x[0] if len(x) == 1 else x

            file_objs = func(lst_objs)

            self.log_writer.start_log(
                "exit", self.class_name, method_name,
            )

            return file_objs

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name,
            )

    def load_model(self, model_name, bucket, log_file, model_dir=None):
        """
        Method Name :   load_model
        Description :   This method loads the model from s3 bucket

        Output      :   A pandas series object consisting of runs for the particular experiment id
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.load_model.__name__

        self.log_writer.start_log(
            "start", self.class_name, method_name,
        )

        try:
            func = (
                lambda: model_name + self.file_format
                if model_dir is None
                else model_dir + model_name + self.file_format
            )

            model_file = func()

            self.log_writer.log(
                log_file == log_file, log_file, f"Got {model_file} as model file",
            )

            f_obj = self.get_file_object(model_name, bucket, log_file)

            model_obj = self.read_object(f_obj, decode=False)

            model = pickle.loads(model_obj)

            self.log_writer.log(
                log_file, f"Loaded {model_name} from bucket {bucket}",
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name,
            )

            return model

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name,
            )

    def save_model(self, model, model_dir, model_bucket, log_file, idx=None):
        """
        Method Name :   save_model
        Description :   This method saves the model into particular model directory in s3 bucket with kwargs

        Output      :   A pandas series object consisting of runs for the particular experiment id
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.save_model.__name__

        self.log_writer.start_log(
            "start", self.class_name, method_name,
        )

        try:
            model_name = self.model_utils.get_model_name(model, log_file)

            func = (
                lambda: model_name + self.file_format
                if model_name == "KMeans"
                else model_name + str(idx) + self.file_format
            )

            model_file = func()

            with open(file=model_file, mode="wb") as f:
                pickle.dump(model, f)

            self.log_writer.log(
                log_file, f"Saved {model_name} model as {model_file} name",
            )

            bucket_model_path = model_dir + "/" + model_file

            self.log_writer.log(
                log_file, f"Uploading {model_file} to {model_bucket} bucket",
            )

            self.upload_file(
                model_file, bucket_model_path, model_bucket,
            )

            self.log_writer.log(
                log_file, f"Uploaded  {model_file} to {model_bucket} bucket",
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name,
            )

        except Exception as e:
            self.log_writer.log(
                log_file, f"Model file {model_name} could not be saved",
            )

            self.log_writer.exception_log(
                e, self.class_name, method_name,
            )

    def upload_df_as_csv(self, data_frame, local_fname, bucket_fname, bucket, log_file):
        """
        Method Name :   upload_df_as_csv
        Description :   This method uploades a dataframe as csv file to s3 bucket

        Output      :   A dataframe is uploaded as csv file to s3 bucket
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.upload_df_as_csv.__name__

        self.log_writer.start_log(
            "start", self.class_name, method_name,
        )

        try:
            data_frame.to_csv(local_fname, index=None, header=True)

            self.log_writer.log(
                log_file, f"Created a local copy of dataframe with name {local_fname}",
            )

            self.upload_file(
                local_fname, bucket_fname, bucket,
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name,
            )

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name,
            )
