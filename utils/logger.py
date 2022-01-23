from datetime import datetime
from phising.mongo_db_operations.mongo_operations import MongoDB_Operation


class App_Logger:
    """
    Description :   This class is used for logging the info to MongoDB

    Version     :   1.2
    Revisions   :   moved to setup to cloud
    """

    def __init__(self):
        self.db_obj = MongoDB_Operation()

        self.class_name = self.__class__.__name__

    def log(self, db_name, collection_name, log_message):
        """
        Method Name :   log
        Description :   This method is used for log the info to MongoDB

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.log.__name__

        try:
            self.now = datetime.now()

            self.date = self.now.date()

            self.current_time = self.now.strftime("%H:%M:%S")

            log = {
                "Log_updated_date": self.now,
                "Log_updated_time": self.current_time,
                "Log_message": log_message,
            }

            self.db_obj.insert_one_record(
                db_name=db_name, collection_name=collection_name, record=log
            )

        except Exception as e:
            error_msg = f"Exception occured in Class : {self.class_name}, Method : {method_name}, Error : {str(e)}"

            raise Exception(error_msg)

    def start_log(self, key, class_name, method_name, db_name, collection_name):
        """
        Method Name :   start_log
        Description :   This method is used for logging the entry or exit of method depending on key value

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        start_method_name = self.start_log.__name__

        try:
            if key == "start":
                self.log(
                    db_name=db_name,
                    collection_name=collection_name,
                    log_message=f" Entered {method_name} of class {class_name}",
                )

            elif key == "exit":
                self.log(
                    db_name=db_name,
                    collection_name=collection_name,
                    log_message=f" Exited {method_name} of class {class_name}",
                )

            else:
                pass

        except Exception as e:
            error_msg = f"Exception occured in Class : {self.class_name}, Method : {start_method_name}, Error : {str(e)}"

            raise Exception(error_msg)

    def raise_exception_log(
        self, error, class_name, method_name, db_name, collection_name
    ):
        """
        Method Name :   raise_exception_log
        Description :   This method is used for logging exception 

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """

        self.start_log(
            key="exit",
            class_name=class_name,
            method_name=method_name,
            db_name=db_name,
            collection_name=collection_name,
        )

        exception_msg = f"Exception occured in Class : {class_name}, Method : {method_name}, Error : {str(error)}"

        self.log(
            db_name=db_name, collection_name=collection_name, log_message=exception_msg
        )

        raise Exception(exception_msg)
