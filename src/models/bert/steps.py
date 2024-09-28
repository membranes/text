"""Module steps.py"""
import logging

import transformers

import src.elements.frames as fr
import src.elements.variable as vr
import src.models.bert.metrics
import src.models.bert.modelling
import src.models.bert.validation
import src.models.bert.parameters
import src.models.bert.structures
import src.models.bert.tokenizer


class Steps:
    """
    The BERT steps.
    """

    def __init__(self, enumerator: dict, archetype: dict, frames: fr.Frames):
        """

        :param enumerator: Code -> tag mapping
        :param archetype: Tag -> code mapping
        :param frames: The data frames for modelling stages, i.e., the
                       training, validating, and testing stages
        """

        # Inputs
        self.__enumerator = enumerator
        self.__archetype = archetype
        self.__frames = frames

        # A set of values for machine learning model development
        self.__variable = vr.Variable()
        self.__variable = self.__variable._replace(
            EPOCHS=4, N_TRAIN=self.__frames.training.shape[0], N_TRIALS=8)

        # Instances
        self.__tokenizer = src.models.bert.tokenizer.Tokenizer()()
        self.__structures = src.models.bert.structures.Structures(
            enumerator=self.__enumerator, variable=self.__variable,
            frames=frames, tokenizer=self.__tokenizer)

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def exc(self):
        """
        a. Training
        b. Validation Details
        c. Testing

        :return:
        """

        training = self.__structures.training()
        validating = self.__structures.validating()

        # Training
        model: transformers.PreTrainedModel = src.models.bert.modelling.Modelling(
            variable = self.__variable, enumerator=self.__enumerator,
            dataloader=training.dataloader).exc()

        # Validation Details
        originals, predictions = src.models.bert.validation.Validation(
            model=model, archetype=self.__archetype,
            dataloader=validating.dataloader).exc()

        src.models.bert.metrics.Metrics().exc(
            originals=originals, predictions=predictions)
