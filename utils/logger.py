from logging import DEBUG, basicConfig, error, info
from os import makedirs
from os.path import join


class App_Logger:
    """
    Description :   This class is used for logging the info
    
    Version     :   1.2
    Revisions   :   Moved to setup to cloud 
    """

    def __init__(self):
        self.class_name = self.__class__.__name__

        makedirs("app_logs", exist_ok=True)

    def log(self, log_info: str, log_file):
        try:
            log_file_path = join("app_logs", log_file)

            basicConfig(
                filename=log_file_path,
                level=DEBUG,
                format="%(asctime)s %(levelname)s %(message)s",
                datefmt="%d-%m-%Y %H:%M:%S",
            )

            info(msg=log_info)

        except Exception as e:
            raise e

    def start_log(self, key: str, class_name: str, method_name: str, log_file):
        """
        Method Name :   start_log
        Description :   This method creates an entry point log in DynamoDB

        Output      :   An entry point is created in DynamoDB
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """

        start_method_name = self.start_log.__name__

        try:
            func = lambda: "Entered" if key == "start" else "Exited"

            log_msg = f"{func()} {method_name} method of class {class_name}"

            self.log(log_msg, log_file)

        except Exception as e:
            error_msg = f"Exception occured in Class : {self.class_name}, Method : {start_method_name}, Error : {str(e)}"

            raise Exception(error_msg)

    def exception_log(
        self, exception: str, class_name: str, method_name: str, log_file
    ):
        """
        Method Name :   exception_log
        Description :   This method creates an exception log in DynamoDB and raises Exception

        Output      :   A exception log is created in DynamoDB and expection is raised
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """

        self.start_log("exit", class_name, method_name, log_file)

        exception_msg = f"Exception occured in Class : {class_name}, Method : {method_name}, Error : {str(exception)}"

        log_file_path = join("app_logs", log_file)

        basicConfig(
            filename=log_file_path,
            level=DEBUG,
            format="%(asctime)s %(levelname)s %(message)s",
            datefmt="%d-%m-%Y %H:%M:%S",
        )

        error(exception_msg)

        raise Exception(exception_msg)
