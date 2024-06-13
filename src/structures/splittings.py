import logging
import typing

import pandas as pd

import config


class Initial:

    def __init__(self, frame: pd.DataFrame) -> None:
        """

        :param frame: The data set for the training and validating stages
        """

        self.__frame = frame

        configurations = config.Config()
        self.__fraction = configurations.fraction
        self.__seed = configurations.seed

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def __split(self) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
        """
        This method splits  data set into training & validating data sets.

        :return:
        training : pandas.DataFrame<br>
            The data set for training<br>
        validating : pandas.DataFrame<br>
            The data set for the validating stage
        """

        blob = self.__frame.copy()

        training = blob.sample(frac=self.__fraction, random_state=self.__seed)
        validating = blob.drop(training.index)

        training.reset_index(drop=True, inplace=True)
        validating.reset_index(drop=True, inplace=True)

        self.__logger.info('training: %s', training.shape)
        self.__logger.info('validating: %s', validating.shape)

        return training, validating

    def exc(self) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
        """

        :return:
        training: pandas.DataFrame
            The training stage data
        validating: pandas.DataFrame
            The validating stage data
        """

        return self.__split()
