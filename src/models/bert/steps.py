"""Module steps.py"""
import transformers

import src.elements.frames as fr
import src.elements.variable as vr
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
        # best = ...

