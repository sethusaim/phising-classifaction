import os
import sys
from typing import List, Tuple

import pandas as pd

from phising.exception import PhisingException


class PhisingData:
    def __init__(self):
        pass

    def read_csv_from_folder(
        self, folder_name: str
    ) -> List[Tuple[pd.DataFrame, str, str]]:
        try:
            lst: List[Tuple(pd.DataFrame, str, str)] = [
                (
                    pd.read_csv(folder_name + "/" + f),
                    folder_name + "/" + f,
                    f.split("/")[-1],
                )
                for f in os.listdir(folder_name)
            ]

            return lst

        except Exception as e:
            raise PhisingException(e, sys)
