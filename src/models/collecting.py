import logging

import pandas as pd
import torch.utils.data as tu

import src.elements.collecting as cl
import src.elements.variable as vr
import src.models.bert.dataset
import src.models.loadings


class Collecting:
    """
    Collecting
    ----------

    Builds and delivers the data structures per modelling stage
    """

    def __init__(self, enumerator: dict, variable: vr.Variable,
                 training: pd.DataFrame, validating: pd.DataFrame = None,
                 testing: pd.DataFrame = None):
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

    def __structure(self, frame: pd.DataFrame, parameters: dict) -> cl.Collecting:
        """
        self.__logger.info('%s dataset:\n%s', name, dataset.__dict__)
        self.__logger.info('%s dataloader:\n%s', name, dataloader.__dict__)

        :param frame:
        :param parameters:
        :return:
        """

        dataset = src.models.bert.dataset.Dataset(
            frame=frame, variable=self.__variable, enumerator=self.__enumerator)

        dataloader: tu.DataLoader = self.__loadings.exc(
            dataset=dataset, parameters=parameters)

        return cl.Collecting(dataset=dataset, dataloader=dataloader)

    def training_(self) -> cl.Collecting:
        """
        Delivers the training data's Dataset & DataLoader

        :return:
        """

        parameters = {'batch_size': self.__variable.TRAIN_BATCH_SIZE,
                      'shuffle': True, 'num_workers': 0}

        return self.__structure(frame=self.__training, parameters=parameters)

    def validating_(self) -> cl.Collecting:
        """
        Delivers the validation data's Dataset & DataLoader

        :return:
        """

        parameters = {'batch_size': self.__variable.VALID_BATCH_SIZE,
                      'shuffle': True, 'num_workers': 0}

        return self.__structure(frame=self.__validating, parameters=parameters)

    def testing_(self) -> cl.Collecting:
        """
        Delivers the testing data's Dataset & DataLoader

        :return:
        """

        # The modelling parameters
        parameters = {'batch_size': self.__variable.TEST_BATCH_SIZE,
                      'shuffle': True, 'num_workers': 0}

        return self.__structure(frame=self.__testing, parameters=parameters)
