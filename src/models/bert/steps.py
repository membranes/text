import logging

import pandas as pd

import torch.utils.data as tu

import src.elements.variable
import src.models.bert.dataset
import src.models.loadings


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

        # Instances
        self.__loadings = src.models.loadings.Loadings()

        # A set of values for machine learning model development
        self.__variable = src.elements.variable.Variable()
        self.__variable._replace(EPOCHS=2)

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def __structure(self, blob: pd.DataFrame) -> tu.Dataset:
        """

        :param blob:
        :return:
        """

        dataset: tu.Dataset = src.models.bert.dataset.Dataset(blob, self.__variable, self.__enumerator)
        self.__logger.info('frame: %s', blob.shape)
        self.__logger.info('dataset: %s', dataset.__doc__)

        return dataset

    def exc(self):

        self.__logger.info('Training')
        training_dataset = self.__structure(blob=self.__training)
        training_loader = self.__loadings.exc(training_dataset, parameters={
            'batch_size': self.__variable.TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 0})



        self.__logger.info('Validating')
        validating_dataset = self.__structure(blob=self.__validating)
        validating_loader = self.__loadings.exc(validating_dataset, parameters={
            'batch_size': self.__variable.VALID_BATCH_SIZE, 'shuffle': True, 'num_workers': 0})
