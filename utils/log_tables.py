import boto3
from utils.read_params import read_params


class Create_Log_Table:
    """
    Description :   This class shall be used for creating the log tables in DynamoDB
    Written by  :   iNeuron Intelligence
    
    Version     :   1.2
    Revisions   :   Moved to setup to cloud 
    """

    def __init__(self):
        self.config = read_params()

        self.train_tables = list(self.config["train_db_log"].values())

        self.pred_tables = list(self.config["pred_db_log"].values())

        self.db_client = boto3.client("dynamodb")

        self.db_resource = boto3.resource("dynamodb")

        self.class_name = self.__class__.__name__

    def create_log_table(self, table_name):
        """
        Method Name :   create_log_table
        Description :   This method create a log table in DynamoDB

        Output      :   A table is created in DynamoDB with table_name
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.create_log_table.__name__

        try:
            response = self.db_client.list_tables()

            if table_name in response["TableNames"]:
                pass

            else:
                self.db_resource.create_table(
                    TableName=table_name,
                    KeySchema=[
                        {"AttributeName": "Log_updated_date", "KeyType": "HASH"},
                    ],
                    AttributeDefinitions=[
                        {"AttributeName": "Log_updated_date", "AttributeType": "S"},
                    ],
                    ProvisionedThroughput={
                        "ReadCapacityUnits": 1000,
                        "WriteCapacityUnits": 1000,
                    },
                )

        except Exception as e:
            error_msg = f"Exception occured in Class : {self.class_name}, Method : {method_name}, Error : {str(e)}"

            raise Exception(error_msg)

    def generate_log_tables(self, type):
        """
        Method Name :   generate_log_tables
        Description :   This method generates the log tables based on type (train or pred)

        Output      :   Log tables are generated based on the type
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.generate_log_tables.__name__

        try:
            func = lambda: self.train_tables if type == "train" else self.pred_tables

            tables = func()

            for table in tables:
                self.create_log_table(table_name=table)

        except Exception as e:
            error_msg = f"Exception occured in Class : {self.class_name}, Method : {method_name}, Error : {str(e)}"

            raise Exception(error_msg)
