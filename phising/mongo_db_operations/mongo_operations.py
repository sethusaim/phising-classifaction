import json
import os

import pandas as pd
from pymongo import MongoClient
from utils.logger import App_Logger
from utils.read_params import read_params


class MongoDB_Operation:
    """
    Description :   This method is used for all mongodb operations
    Written by  :   iNeuron Intelligence
    
    Version     :   1.2
    Revisions   :   Moved to setup to cloud 
    """

    def __init__(self):
        self.config = read_params()

        self.class_name = self.__class__.__name__

        self.DB_URL = os.environ["MONGODB_URL"]

        self.client = MongoClient(self.DB_URL)

        self.log_writer = App_Logger()

    def get_database(self, db_name, log_file):
        """
        Method Name :   get_database
        Description :   This method gets database from MongoDB from the db_name

        Output      :   A database is created in MongoDB with name as db_name
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_database.__name__

        self.log_writer.start_log(
            "start", self.class_name, method_name,
        )

        try:
            db = self.client[db_name]

            self.log_writer.log(
                log_file, f"Created {db_name} database in MongoDB",
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name,
            )

            return db

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name,
            )

    def get_collection(self, database, collection_name, log_file):
        """
        Method Name :   get_collection
        Description :   This method gets collection from the particular database and collection name

        Output      :   A collection is returned from database with name as collection name
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_collection.__name__

        self.log_writer.start_log(
            "start", self.class_name, method_name,
        )

        try:
            collection = database[collection_name]

            self.log_writer.log(
                log_file, f"Created {collection_name} collection in mongodb",
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name,
            )

            return collection

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name,
            )

    def get_collection_as_dataframe(self, db_name, collection_name, log_file):
        """
        Method Name :   get_collection_as_dataframe
        Description :   This method is used for converting the selected collection to dataframe

        Output      :   A collection is returned from the selected db_name and collection_name
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Written by  :   iNeuron Intelligence
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_collection_as_dataframe.__name__

        self.log_writer.start_log(
            "start", self.class_name, method_name,
        )

        try:
            database = self.get_database(db_name=db_name, log_file)

            collection = database.get_collection(name=collection_name)

            df = pd.DataFrame(list(collection.find()))

            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)

            self.log_writer.log(
                log_file, "Converted collection to dataframe",
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name,
            )

            return df

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name,
            )

    def insert_dataframe_as_record(
        self, data_frame, db_name, collection_name, log_file
    ):
        """
        Method Name :   insert_dataframe_as_record
        Description :   This method inserts the dataframe as record in database collection

        Output      :   The dataframe is inserted in database collection
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.insert_dataframe_as_record.__name__

        self.log_writer.start_log(
            "start", self.class_name, method_name,
        )

        try:
            records = json.loads(data_frame.T.to_json()).values()

            self.log_writer.log(
                log_file, f"Converted dataframe to json records",
            )

            database = self.get_database(db_name, log_file)

            collection = database.get_collection(collection_name)

            self.log_writer.log(log_file, "Inserting records to MongoDB")

            collection.insert_many(records)

            self.log_writer.log(log_file, "Inserted records to MongoDB")

            self.log_writer.start_log(
                "exit", self.class_name, method_name,
            )

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name,
            )
