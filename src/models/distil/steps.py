"""Module steps.py"""
import os
import logging

import src.elements.frames as fr
import src.elements.variable as vr
import src.models.distil.intelligence
import src.models.distil.parameters
import src.models.distil.validation
import src.models.structures


class Steps:
    """
    Steps
    """

    def __init__(self, enumerator: dict, archetype: dict, frames: fr.Frames):
        """

        :param enumerator:
        :param archetype:
        :param frames:
        """

        # Inputs
        self.__enumerator = enumerator
        self.__archetype = archetype

        # A set of values for machine learning model development
        self.__variable = vr.Variable()
        self.__variable = self.__variable._replace(EPOCHS=8, TRAIN_BATCH_SIZE=16, VALID_BATCH_SIZE=16)

        # Instances
        parameters = src.models.distil.parameters.Parameters()
        self.__structures = src.models.structures.Structures(
            enumerator=self.__enumerator, variable=self.__variable,
            frames=frames, tokenizer=parameters.tokenizer)

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def exc(self):
        """

        :return:
        """

        # The data
        training = self.__structures.training()
        validating = self.__structures.validating()

        # Modelling
        intelligence = src.models.distil.intelligence.Intelligence(
            variable=self.__variable, enumerator=self.__enumerator)
        model = intelligence(training=training, validating=validating)

        # Evaluating
        src.models.distil.validation.Validation(
            validating=validating).exc(model=model)
