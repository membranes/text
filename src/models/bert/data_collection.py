import logging
import typing

import pandas as pd

import torch.utils.data as tu

import src.elements.variable
import src.models.bert.dataset
import src.models.loadings

class DataCollection:

    def __init__(self, enumerator: dict, variable: src.elements.variable.Variable):
        """

        :param enumerator:
        """

        self.__enumerator = enumerator

        # For DataLoader creation
        self.__loadings = src.models.loadings.Loadings()

        # A set of values for machine learning model development
        self.__variable = variable

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def exc(self, blob: pd.DataFrame, parameters: dict) -> typing.Tuple[tu.Dataset, tu.DataLoader]:
        """

        :param blob:
        :param parameters:
        :return:
        """

        dataset: tu.Dataset = src.models.bert.dataset.Dataset(
            frame=blob, variable=self.__variable, enumerator=self.__enumerator)
        self.__logger.info('frame: %s', blob.shape)
        self.__logger.info('dataset: %s', dataset.__doc__)

        dataloader: tu.DataLoader = self.__loadings.exc(
            dataset=dataset, parameters=parameters)

        return dataset, dataloader
