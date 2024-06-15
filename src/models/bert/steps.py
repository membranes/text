import logging

import pandas as pd
import transformers

import src.elements.variable
import src.models.bert.data_collection
import src.models.bert.dataset
import src.models.loadings
import src.models.bert.modelling
import src.models.bert.validation
import src.models.metrics


class Steps:

    def __init__(self, enumerator: dict, archetype: dict,
                 training: pd.DataFrame, validating: pd.DataFrame):
        """

        :param enumerator:
        :param archetype:
        :param training:
        :param validating:
        """

        # Inputs
        self.__enumerator = enumerator
        self.__archetype = archetype
        self.__training = training
        self.__validating = validating

        # A set of values for machine learning model development
        self.__variable = src.elements.variable.Variable()
        self.__variable = self.__variable._replace(EPOCHS=2)

        # Instances
        self.__loadings = src.models.loadings.Loadings()
        self.__data_collection = src.models.bert.data_collection.DataCollection(
            enumerator=self.__enumerator, variable=self.__variable)

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def exc(self):
        """

        :return:
        """

        self.__logger.info('Training Data')
        _, training_dataloader = self.__data_collection.exc(blob=self.__training, parameters={
            'batch_size': self.__variable.TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}, name='training')

        self.__logger.info('Validation Data')
        _, validating_dataloader = self.__data_collection.exc(blob=self.__validating, parameters={
            'batch_size': self.__variable.VALID_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}, name='validating')

        self.__logger.info('Modelling: Training Stage')
        model: transformers.PreTrainedModel = src.models.bert.modelling.Modelling(
            variable = self.__variable, enumerator=self.__enumerator, dataloader=training_dataloader).exc()

        self.__logger.info('Modelling: Validation Stage')
        originals, predictions = src.models.bert.validation.Validation(model=model, dataloader=validating_dataloader,
                                                                       archetype=self.__archetype).exc()
        src.models.metrics.Metrics().exc(originals=originals, predictions=predictions)
