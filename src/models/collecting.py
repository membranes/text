import logging
import typing

import pandas as pd
import torch.utils.data as tu

import src.elements.variable as vr
import src.elements.collecting as cl
import src.models.bert.dataset
import src.models.loadings


class Collecting:

    def __init__(self, enumerator: dict, variable: vr.Variable,
                 training: pd.DataFrame, validating: pd.DataFrame = None, testing: pd.DataFrame = None):
        """

        :param enumerator:
        :param variable:
        :param training:
        :param validating:
        :param testing:
        """

        # A set of values for machine learning model development
        self.__enumerator = enumerator
        self.__variable = variable
        self.__training = training
        self.__validating = validating
        self.__testing = testing

        # For DataLoader creation
        self.__loadings = src.models.loadings.Loadings()

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def training_(self) -> cl.Collecting:

        parameters = {'batch_size': self.__variable.TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}

        dataset: tu.Dataset = src.models.bert.dataset.Dataset(
            frame=self.__training, variable=self.__variable, enumerator=self.__enumerator)

        dataloader: tu.DataLoader = self.__loadings.exc(dataset=dataset, parameters=parameters)

        return cl.Collecting(dataset=dataset, dataloader=dataloader)

    def validating_(self) -> cl.Collecting:

        parameters = {'batch_size': self.__variable.VALID_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}

        dataset: tu.Dataset = src.models.bert.dataset.Dataset(
            frame=self.__validating, variable=self.__variable, enumerator=self.__enumerator)

        dataloader: tu.DataLoader = self.__loadings.exc(dataset=dataset, parameters=parameters)

        return cl.Collecting(dataset=dataset, dataloader=dataloader)

    def testing_(self) -> cl.Collecting:

        parameters = {'batch_size': self.__variable.TEST_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}

        dataset: tu.Dataset = src.models.bert.dataset.Dataset(
            frame=self.__testing, variable=self.__variable, enumerator=self.__enumerator)

        dataloader: tu.DataLoader = self.__loadings.exc(dataset=dataset, parameters=parameters)

        return cl.Collecting(dataset=dataset, dataloader=dataloader)

    def exc(self, blob: pd.DataFrame, parameters: dict, name: str = None) -> typing.Tuple[tu.Dataset, tu.DataLoader]:
        """
        self.__logger.info('%s dataset:\n%s', name, dataset.__dict__)
        self.__logger.info('%s dataloader:\n%s', name, dataloader.__dict__)

        :param blob: The dataframe being transformed
        :param parameters: The modelling parameters of <blob>
        :param name: A descriptive name, e.g., training, validating, etc.
        :return:
        """

        dataset: tu.Dataset = src.models.bert.dataset.Dataset(
            frame=blob, variable=self.__variable, enumerator=self.__enumerator)

        dataloader: tu.DataLoader = self.__loadings.exc(
            dataset=dataset, parameters=parameters)

        return dataset, dataloader
