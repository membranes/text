"""Module steps.py"""
import logging
import transformers

import src.elements.frames as fr
import src.elements.variable as vr
import src.models.bert.architecture
import src.models.bert.measurements
import src.models.bert.operating
import src.models.bert.parameters
import src.models.bert.structures
import src.models.bert.tokenizer
import src.models.bert.validation


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
        self.__tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase = (
            src.models.bert.tokenizer.Tokenizer()())

    def __structures(self):
        """

        :return:
        """

        structures = src.models.bert.structures.Structures(
            enumerator=self.__enumerator, variable=self.__variable,
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
            variable=self.__variable, enumerator=self.__enumerator, archetype=self.__archetype
        )
        best = architecture(training=training, validating=validating, tokenizer=self.__tokenizer)
        logging.info(best)

        # Hence, update the modelling variables
        self.__variable = self.__variable._replace(
            LEARNING_RATE=best.hyperparameters.get('learning_rate'), WEIGHT_DECAY=best.hyperparameters.get('weight_decay'))
        logging.info(self.__variable)

        # Training via the best hyperparameters set
        operating = src.models.bert.operating.Operating(
            variable=self.__variable, enumerator=self.__enumerator, archetype=self.__archetype
        )
        model = operating.exc(training=training, validating=validating, tokenizer=self.__tokenizer)
        logging.info(dir(model))

        # Evaluating: vis-à-vis model & validation data
        originals, predictions = src.models.bert.validation.Validation(
            validating=validating, archetype=self.__archetype).exc(model=model)





