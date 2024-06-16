"""Module steps.py"""
import logging

import pandas as pd
import transformers

import src.elements.variable
import src.models.structures
import src.models.bert.dataset
import src.models.loadings
import src.models.bert.modelling
import src.models.bert.validation
import src.models.metrics


class Steps:
    """
    The BERT steps.
    """

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

        # A set of values for machine learning model development
        self.__variable = src.elements.variable.Variable()
        self.__variable = self.__variable._replace(EPOCHS=2)

        # Instances
        self.__loadings = src.models.loadings.Loadings()
        self.__structures = src.models.structures.Structures(
            enumerator=self.__enumerator, variable=self.__variable,
            training=training, validating=validating)

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def exc(self):
        """

        :return:
        """

        training = self.__structures.training()
        validating = self.__structures.validating()

        self.__logger.info('Modelling: Training Stage')
        model: transformers.PreTrainedModel = src.models.bert.modelling.Modelling(
            variable = self.__variable, enumerator=self.__enumerator,
            dataloader=training.dataloader).exc()

        self.__logger.info('Modelling: Validation Stage')
        originals, predictions = src.models.bert.validation.Validation(
            model=model, archetype=self.__archetype,
            dataloader=validating.dataloader).exc()

        self.__logger.info('Metrics')
        src.models.metrics.Metrics().exc(originals=originals, predictions=predictions)
