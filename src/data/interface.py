"""Module interface.py"""
import dask.dataframe as dfr
import pandas as pd

import src.elements.s3_parameters as s3p
import config


class Interface:
    """
    Class Interface
    """

    def __init__(self, s3_parameters: s3p.S3Parameters):
        """

        :param s3_parameters: s3_parameters: The overarching S3 (Simple Storage Service) parameters
                              settings of this project, e.g., region code name, buckets, etc.
        """

        self.__s3_parameters = s3_parameters

        # Configurations
        self.__configurations = config.Config()

    def data(self) -> pd.DataFrame:
        """

        :return:
        """

        path = ('s3://' + self.__s3_parameters.internal + '/' +
                self.__s3_parameters.path_internal_data + self.__configurations.data_)

        try:
            frame: dfr.DataFrame = dfr.read_csv(path=path, header=0)
        except ImportError as err:
            raise err from err

        return frame.compute()
