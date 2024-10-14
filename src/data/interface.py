import dask.dataframe as dfr
import pandas as pd

import src.elements.s3_parameters as s3p
import config


class Interface:

    def __init__(self, s3_parameters: s3p.S3Parameters):

        self.__s3_parameters = s3_parameters

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
