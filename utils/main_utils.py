from os import listdir
from os.path import join
from shutil import rmtree

from phising.s3_bucket_operations.s3_operations import S3_Operation

from utils.logger import App_Logger
from utils.read_params import read_params


class Main_Utils:
    def __init__(self):
        self.s3 = S3_Operation()

        self.log_writer = App_Logger()

        self.config = read_params()

        self.inputs_files_bucket = self.config["s3_bucket"]["input_files"]

        self.class_name = self.__class__.__name__

        self.log_file = self.config["upload_log"]

    def upload_logs(self):
        method_name = self.upload_logs.__name__

        self.log_writer.start_log("start", self.class_name, method_name, self.log_file)

        try:
            lst = listdir("logs")

            self.log_writer.log("Got list of logs from logs folder", self.log_file)

            for f in lst:
                local_f = join("logs", f)

                dest_f = "logs" + "/" + f

                self.s3.upload_file(
                    local_f, dest_f, self.inputs_files_bucket, self.log_file
                )

            self.log_writer.log(
                f"Uploaded logs to {self.inputs_files_bucket}", self.log_file
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name, self.log_file
            )

            rmtree("logs")

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name, self.log_file
            )
