"""Module steps.py"""
import logging

import transformers

import src.elements.arguments as ag
import src.elements.frames as fr
import src.models.bert.architecture
import src.models.bert.operating
import src.models.bert.parameters
import src.models.bert.tokenizer
import src.models.bert.validation
import src.models.measurements
import src.models.bert.structures


class Steps:
    """
    The BERT steps.
    """

    def __init__(self, enumerator: dict, archetype: dict, arguments: ag.Arguments, frames: fr.Frames):
        """

        :param enumerator: Code -> tag mapping
        :param archetype: Tag -> code mapping
        :param arguments: The parameter values for ...
        :param frames: The data frames for modelling stages, i.e., the
                       training, validating, and testing stages
        """

        # Inputs
        self.__enumerator = enumerator
        self.__archetype = archetype
        self.__arguments = arguments
        self.__frames = frames

        # A set of values for machine learning model development
        self.__arguments = self.__arguments._replace(
            N_TRAIN=self.__frames.training.shape[0], N_VALID=self.__frames.validating.shape[0], N_TEST=self.__frames.testing.shape[0])

        # Get tokenizer
        self.__tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase = (
            src.models.bert.tokenizer.Tokenizer()())

    def __structures(self):
        """

        :return:
        """

        structures = src.models.bert.structures.Structures(
            enumerator=self.__enumerator, arguments=self.__arguments,
            frames=self.__frames, tokenizer=self.__tokenizer)

        return structures.training(), structures.validating(), structures.testing()

    def exc(self):
        """
        a. Training
        b. Evaluating
        c. Testing

        :return:
        """

        training, validating, _ = self.__structures()

        # Hyperparameter search
        architecture = src.models.bert.architecture.Architecture(
            arguments=self.__arguments, enumerator=self.__enumerator, archetype=self.__archetype)
        best = architecture(training=training, validating=validating, tokenizer=self.__tokenizer)
        logging.info(best)

        # Hence, update the modelling variables
        self.__arguments = self.__arguments._replace(
            LEARNING_RATE=best.hyperparameters.get('learning_rate'), WEIGHT_DECAY=best.hyperparameters.get('weight_decay'))
        logging.info(self.__arguments)

        # Training via the best hyperparameters set
        operating = src.models.bert.operating.Operating(
            arguments=self.__arguments, enumerator=self.__enumerator, archetype=self.__archetype)
        model = operating.exc(training=training, validating=validating, tokenizer=self.__tokenizer)
        logging.info(dir(model))

        # Evaluating: vis-à-vis model & validation data
        originals, predictions = src.models.bert.validation.Validation(
            validating=validating, archetype=self.__archetype).exc(model=model)

        src.models.measurements.Measurements().exc(
            originals=originals, predictions=predictions)
