from datetime import datetime
import boto3


class App_Logger:
    """
    Description :   This class is used for logging the info to DynamoDB

    Version     :   1.2
    Revisions   :   moved to setup to cloud
    """

    def __init__(self):
        self.db_resource = boto3.resource("dynamodb")

        self.class_name = self.__class__.__name__

    def log(self, table_name, log_message):
        """
        Method Name :   log
        Description :   This method is used for log the info to DynamoDB

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.log.__name__

        try:
            self.table = self.db_resource.Table(table_name)

            self.now = datetime.now()

            self.date = self.now.date()

            self.current_time = self.now.strftime("%H:%M:%S")

            log = {
                "Log_updated_date": str(self.now),
                "Log_updated_time": str(self.current_time),
                "Log_message": log_message,
            }

            self.table.put_item(Item=log)

        except Exception as e:
            error_msg = f"Exception occured in Class : {self.class_name}, Method : {method_name}, Error : {str(e)}"

            raise Exception(error_msg)

    def start_log(self, key, class_name, method_name, table_name):
        """
        Method Name :   start_log
        Description :   This method is used for logging the entry or exit of method depending on key value

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        start_method_name = self.start_log.__name__

        try:
            func = lambda: "Entered" if key == "start" else "Exited"

            log_msg = f"{func()} {method_name} method of class {class_name}"

            self.log(table_name, log_message=log_msg)

        except Exception as e:
            error_msg = f"Exception occured in Class : {self.class_name}, Method : {start_method_name}, Error : {str(e)}"

            raise Exception(error_msg)

    def exception_log(self, error, class_name, method_name, table_name):
        """
        Method Name :   exception_log
        Description :   This method is used for logging exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """

        self.start_log(
            key="exit",
            class_name=class_name,
            method_name=method_name,
            table_name=table_name,
        )

        exception_msg = f"Exception occured in Class : {class_name}, Method : {method_name}, Error : {str(error)}"

        self.log(table_name=table_name, log_message=exception_msg)

        raise Exception(exception_msg)
