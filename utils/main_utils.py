import json
import pickle
from io import StringIO

import pandas as pd

from utils.logger import App_Logger
from utils.read_params import read_params

file_name = "main_utils.py"

config = read_params()

log_writer = App_Logger()


def make_readable(data, db_name, collection_name):
    """
    Method Name :   make_readable
    Description :   This method is used for converting the bytes object to string data

    Version     :   1.2
    Revisions   :   moved setup to cloud
    """
    method_name = make_readable.__name__

    log_writer.start_log(
        key="start",
        class_name=file_name,
        method_name=method_name,
        db_name=db_name,
        collection_name=collection_name,
    )

    try:
        f = StringIO(data)

        log_writer.log(
            db_name=db_name,
            collection_name=collection_name,
            log_message="Converted bytes content to string content using StringIO",
        )

        log_writer.start_log(
            key="exit",
            class_name=file_name,
            method_name=method_name,
            db_name=db_name,
            collection_name=collection_name,
        )

        return f

    except Exception as e:
        log_writer.raise_exception_log(
            error=e,
            file_name=file_name,
            method_name=method_name,
            db_name=db_name,
            collection_name=collection_name,
        )


def convert_object_to_dataframe(obj, db_name, collection_name):
    """
    Method Name :   convert_object_to_dataframe
    Description :   This method is used for converting the s3 object to dataframe

    Version     :   1.2
    Revisions   :   moved setup to cloud
    """
    method_name = convert_object_to_dataframe.__name__

    log_writer.start_log(
        key="start",
        class_name=file_name,
        method_name=method_name,
        db_name=db_name,
        collection_name=collection_name,
    )

    try:
        content = convert_object_to_bytes(
            obj, db_name=db_name, collection_name=collection_name
        )

        data = make_readable(
            data=content, db_name=db_name, collection_name=collection_name
        )

        df = pd.read_csv(data)

        log_writer.log(
            db_name=db_name,
            collection_name=collection_name,
            log_message=f"Converted {obj} to dataframe",
        )

        log_writer.start_log(
            key="start",
            class_name=file_name,
            method_name=method_name,
            db_name=db_name,
            collection_name=collection_name,
        )

        return df

    except Exception as e:
        log_writer.raise_exception_log(
            error=e,
            file_name=file_name,
            method_name=method_name,
            db_name=db_name,
            collection_name=collection_name,
        )


def read_s3_obj(obj, db_name, collection_name, decode=True):
    """
    Method Name :   read_s3_obj
    Description :   This method is used for reading a object from s3 bucket

    Version     :   1.2
    Revisions   :   moved setup to cloud
    """
    method_name = read_s3_obj.__name__

    log_writer.start_log(
        key="start",
        class_name=file_name,
        method_name=method_name,
        db_name=db_name,
        collection_name=collection_name,
    )

    try:
        if decode:
            content = obj.get()["Body"].read().decode()

        else:
            content = obj.get()["Body"].read()

        log_writer.log(
            db_name=db_name,
            collection_name=collection_name,
            log_message=f"Read the object with decode as {decode}",
        )

        log_writer.start_log(
            key="exit",
            class_name=file_name,
            method_name=method_name,
            db_name=db_name,
            collection_name=collection_name,
        )

        return content

    except Exception as e:
        log_writer.raise_exception_log(
            error=e,
            class_name=file_name,
            method_name=method_name,
            db_name=db_name,
            collection_name=collection_name,
        )


def convert_object_to_pickle(obj, db_name, collection_name):
    """
    Method Name :   convert_object_to_pickle
    Description :   This method is used for converting the s3 obj to pickle format

    Version     :   1.2
    Revisions   :   moved setup to cloud
    """
    method_name = convert_object_to_pickle.__name__

    log_writer.start_log(
        key="start",
        class_name=file_name,
        method_name=method_name,
        db_name=db_name,
        collection_name=collection_name,
    )

    try:
        model_content = read_s3_obj(
            obj, decode=False, db_name=db_name, collection_name=collection_name,
        )

        model = pickle.loads(model_content)

        log_writer.log(
            db_name=db_name,
            collection_name=collection_name,
            log_message=f"Loaded {obj} as pickle model",
        )

        log_writer.start_log(
            key="start",
            class_name=file_name,
            method_name=method_name,
            db_name=db_name,
            collection_name=collection_name,
        )

        return model

    except Exception as e:
        log_writer.raise_exception_log(
            error=e,
            file_name=file_name,
            method_name=method_name,
            db_name=db_name,
            collection_name=collection_name,
        )


def convert_object_to_bytes(obj, db_name, collection_name):
    """
    Method Name :   convert_object_to_bytes
    Description :   This method is used for converting the s3 object to bytes

    Version     :   1.2
    Revisions   :   moved setup to cloud
    """
    method_name = convert_object_to_bytes.__name__

    log_writer.start_log(
        key="start",
        class_name=file_name,
        method_name=method_name,
        db_name=db_name,
        collection_name=collection_name,
    )

    try:
        content = read_s3_obj(
            obj, decode=True, db_name=db_name, collection_name=collection_name,
        )

        log_writer.log(
            db_name=db_name,
            collection_name=collection_name,
            log_message=f"Converted {obj} to bytes",
        )

        log_writer.start_log(
            key="start",
            class_name=file_name,
            method_name=method_name,
            db_name=db_name,
            collection_name=collection_name,
        )

        return content

    except Exception as e:
        log_writer.raise_exception_log(
            error=e,
            file_name=file_name,
            method_name=method_name,
            db_name=db_name,
            collection_name=collection_name,
        )


def convert_obj_to_json(obj, db_name, collection_name):
    """
    Method Name :   convert_obj_to_json
    Description :   This method is used for converting the s3 object to json 

    Version     :   1.2
    Revisions   :   moved setup to cloud
    """
    method_name = convert_obj_to_json.__name__

    log_writer.start_log(
        key="start",
        class_name=file_name,
        method_name=method_name,
        db_name=db_name,
        collection_name=collection_name,
    )

    try:
        res = convert_object_to_bytes(
            obj=obj, db_name=db_name, collection_name=collection_name
        )

        dic = json.loads(res)

        log_writer.log(
            db_name=db_name,
            collection_name=collection_name,
            log_message=f"Converted {obj} to json",
        )

        log_writer.start_log(
            key="start",
            class_name=file_name,
            method_name=method_name,
            db_name=db_name,
            collection_name=collection_name,
        )

        return dic

    except Exception as e:
        log_writer.raise_exception_log(
            error=e,
            file_name=file_name,
            method_name=method_name,
            db_name=db_name,
            collection_name=collection_name,
        )
