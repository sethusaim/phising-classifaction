import json
import os

import pandas as pd
from pymongo import MongoClient
from utils.logger import App_Logger
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

        self.client = MongoClient(self.DB_URL)

        self.log_writer = App_Logger()

    def create_db(self, db_name, table_name):
        """
        Method Name :   create_db
        Description :   This method is creating a database in MongoDB

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.create_db.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            db = self.client[db_name]

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Created {db_name} database in MongoDB",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

            return db

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def get_collection(self, database, collection_name, table_name):
        """
        Method Name :   create_collection
        Description :   This method is used for creating a collection in created database

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.create_collection.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            collection = database[collection_name]

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Created {collection_name} collection in mongodb",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

            return collection

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def get_collection_as_dataframe(self, db_name, collection_name, table_name):
        """
        Method Name :   get_collection_as_dataframe
        Description :   This method is used for converting the selected collection to dataframe

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_collection_as_dataframe.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            database = self.create_db(self.client, db_name, table_name=table_name)

            collection = database.get_collection(name=collection_name)

            df = pd.DataFrame(list(collection.find()))

            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)

            self.log_writer.log(
                table_name=table_name, log_message="Converted collection to dataframe",
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

    def insert_dataframe_as_record(
        self, data_frame, db_name, collection_name, table_name
    ):
        """
        Method Name :   insert_dataframe_as_record
        Description :   This method is used for inserting the dataframe in collection as record

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.insert_dataframe_as_record.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            records = json.loads(data_frame.T.to_json()).values()

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Converted dataframe to json records",
            )

            database = self.create_db(db_name, table_name=table_name)

            collection = database.get_collection(collection_name)

            self.log_writer.log(
                table_name=table_name, log_message="Inserting records to MongoDB"
            )

            collection.insert_many(records)

            self.log_writer.log(
                table_name=table_name, log_message="Inserted records to MongoDB"
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
