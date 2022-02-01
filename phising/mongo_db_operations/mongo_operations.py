import json
import os

import pandas as pd
from pymongo import MongoClient
from utils.read_params import read_params


class MongoDB_Operation:
    """
    Description :   This method is used for all mongodb operations  

    Version     :   1.2
    Revisions   :   moved to setup to cloud
    """

    def __init__(self):
        self.config = read_params()

        self.class_name = self.__class__.__name__

        self.DB_URL = os.environ["MONGODB_URL"]

    def get_client(self):
        """
        Method Name :   get_client
        Description :   This method is used for getting the client from MongoDB

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_client.__name__

        try:
            self.client = MongoClient(self.DB_URL)

            return self.client

        except Exception as e:
            exception_msg = f"Exception occured in Class : {self.class_name}, Method : {method_name}, Error : {str(e)}"

            raise Exception(exception_msg)

    def create_db(self, client, db_name):
        """
        Method Name :   create_db
        Description :   This method is creating a database in MongoDB

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.create_db.__name__

        try:
            return client[db_name]

        except Exception as e:
            exception_msg = f"Exception occured in Class : {self.class_name}, Method : {method_name}, Error : {str(e)}"

            raise Exception(exception_msg)

    def create_collection(self, database, collection_name):
        """
        Method Name :   create_collection
        Description :   This method is used for creating a collection in created database

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.create_collection.__name__

        try:
            return database[collection_name]

        except Exception as e:
            exception_msg = f"Exception occured in Class : {self.class_name}, Method : {method_name}, Error : {str(e)}"

            raise Exception(exception_msg)

    def get_collection(self, collection_name, database):
        """
        Method Name :   get_collection
        Description :   This method is used for getting collection from a database

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_collection.__name__

        try:
            collection = self.create_collection(database, collection_name)

            return collection

        except Exception as e:
            exception_msg = f"Exception occured in Class : {self.class_name}, Method : {method_name}, Error : {str(e)}"

            raise Exception(exception_msg)

    def convert_collection_to_dataframe(self):
        """
        Method Name :   convert_collection_to_dataframe
        Description :   This method is used for converting the selected collection to dataframe

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.convert_collection_to_dataframe.__name__

        try:
            client = self.get_client()

            database = self.create_db(client, db_name)

            collection = database.get_collection(name=collection_name)

            df = pd.DataFrame(list(collection.find()))

            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)

            return df

        except Exception as e:
            exception_msg = f"Exception occured in Class : {self.class_name}, Method : {method_name}, Error : {str(e)}"

            raise Exception(exception_msg)

    def is_record_present(self, record):
        """
        Method Name :   is_record_present
        Description :   This method is used for checking whether the record exists or not 

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.is_record_present.__name__

        try:
            client = self.get_client()

            database = self.create_db(client, db_name)

            collection = self.get_collection(collection_name, database)

            record_count = collection.count_documents(filter=record)

            if record_count > 0:
                client.close()

                return True

            else:
                client.close()

                return False

        except Exception as e:
            client.close()

            exception_msg = f"Exception occured in Class : {self.class_name}, Method : {method_name}, Error : {str(e)}"

            raise Exception(exception_msg)

    def create_one_record(self, collection, data):
        """
        Method Name :   create_one_record
        Description :   This method is used for creating a single record in the collection

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.create_one_record.__name__

        try:
            collection.insert_one(data)

            return 1

        except Exception as e:
            exception_msg = f"Exception occured in Class : {self.class_name}, Method : {method_name}, Error : {str(e)}"

            raise Exception(exception_msg)

    def insert_dataframe_as_record(self, data_frame):
        """
        Method Name :   insert_dataframe_as_record
        Description :   This method is used for inserting the dataframe in collection as record

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.insert_dataframe_as_record.__name__

        try:
            records = json.loads(data_frame.T.to_json()).values()

            client = self.get_client()

            database = self.create_db(client, db_name)

            collection = database.get_collection(collection_name)

            collection.insert_many(records)

        except Exception as e:
            exception_msg = f"Exception occured in Class : {self.class_name}, Method : {method_name}, Error : {str(e)}"

            raise Exception(exception_msg)

    def insert_one_record(self, record):
        """
        Method Name :   insert_one_record
        Description :   This method is used for inserting one record into a collection

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.insert_one_record.__name__

        try:
            client = self.get_client()

            database = self.create_db(client, db_name)

            collection = self.get_collection(collection_name, database)

            if not self.is_record_present(db_name, collection_name, record):
                self.create_one_record(collection=collection, data=record)

            client.close()

        except Exception as e:
            exception_msg = f"Exception occured in Class : {self.class_name}, Method : {method_name}, Error : {str(e)}"

            raise Exception(exception_msg)
