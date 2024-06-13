import logging

import pandas as pd

import src.elements.variable
import src.structures.bert.dataset


class Steps:

    def __init__(self, enumerator: dict, training: pd.DataFrame, validating: pd.DataFrame):
        """

        :param enumerator:
        :param training:
        :param validating:
        """

        # Inputs
        self.__enumerator = enumerator
        self.__training = training
        self.__validating = validating

        # A set of values for machine learning model development
        self.__variable = src.elements.variable.Variable()
        self.__variable._replace(EPOCHS=2)

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def __structure(self, blob: pd.DataFrame) -> dict:

        dataset = src.structures.bert.dataset.Dataset(blob, self.__variable, self.__enumerator)
        self.__logger.info('frame: %s', blob.shape)
        self.__logger.info('dataset: %s', dataset.__len__())

        return dataset

    def exc(self):

        self.__logger.info('Training')
        self.__structure(blob=self.__training)

        self.__logger.info('Validating')
        self.__structure(blob=self.__validating)
