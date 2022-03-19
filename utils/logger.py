import os
from datetime import datetime


class App_Logger:
    def __init__(self):
        self.class_name = self.__class__.__name__

        os.makedirs("logs", exist_ok=True)

    def log(self, log_file, log_info):
        try:
            self.now = datetime.now()

            self.date = self.now.strftime("%d:%m:%Y")

            self.current_time = self.now.strftime("%H:%M:%S")

            log_file_path = "logs" + "/" + log_file

            with open(file=log_file_path, mode="a+") as f:
                f.write(
                    str(self.date)
                    + "\t"
                    + str(self.current_time)
                    + "\t"
                    + log_info
                    + "\n"
                )

                f.close()

        except Exception as e:
            raise e

    def start_log(self, key, log_file, class_name, method_name):
        start_method_name = self.start_log.__name__

        try:
            func = lambda: "Entered" if key == "start" else "Exited"

            key = func()

            log_msg = f"{key} {method_name} method of class {class_name}"

            self.log(log_file, log_msg)

        except Exception as e:
            error_msg = f"Exception occured in Class : {self.class_name}, Method : {start_method_name}, Error : {str(e)}"

            raise Exception(error_msg)

    def exception_log(self, error, log_file, class_name, method_name):
        self.start_log("exit", log_file, class_name, method_name)

        exception_msg = f"Exception occured in Class : {class_name}, Method : {method_name}, Error : {str(error)}"

        self.log(log_file, exception_msg)

        raise Exception(exception_msg)
