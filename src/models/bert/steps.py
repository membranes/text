"""Module steps.py"""
import logging

import transformers

import src.elements.arguments as ag
import src.elements.hyperspace as hp
import src.elements.vault as vu
import src.models.bert.structures
import src.models.bert.tokenizer
import src.models.optimal
import src.models.prime


class Steps:
    """
    The BERT steps.
    """

    def __init__(self, enumerator: dict, archetype: dict, arguments: ag.Arguments, hyperspace: hp.Hyperspace, vault: vu.Vault):
        """

        :param enumerator: Code -> tag mapping
        :param archetype: Tag -> code mapping
        :param arguments: The parameter values for ...
        :param hyperspace: The real number spaces of ...
        :param vault: The data frames for modelling stages, i.e., the
                       training, validating, and testing stages
        """

        # Inputs
        self.__enumerator = enumerator
        self.__archetype = archetype
        self.__arguments = arguments
        self.__hyperspace = hyperspace
        self.__vault = vault

        # A set of values for machine learning model development
        self.__arguments = self.__arguments._replace(
            N_TRAIN=self.__vault.training.shape[0], N_VALID=self.__vault.validating.shape[0], N_TEST=self.__vault.testing.shape[0])

        # Get tokenizer
        self.__tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase = (
            src.models.bert.tokenizer.Tokenizer(arguments=self.__arguments)())

    def __structures(self):
        """

        :return:
        """

        structures = src.models.bert.structures.Structures(
            enumerator=self.__enumerator, arguments=self.__arguments,
            vault=self.__vault, tokenizer=self.__tokenizer)

        return structures.training(), structures.validating(), structures.testing()

    def exc(self):
        """

        :return:
        """

        training, validating, _ = self.__structures()

        # Hyperparameter search
        optimal = src.models.optimal.Optimal(
            arguments=self.__arguments, hyperspace=self.__hyperspace,
            enumerator=self.__enumerator, archetype=self.__archetype)
        best = optimal(training=training, validating=validating, tokenizer=self.__tokenizer)

        # Hence, update the modelling variables
        self.__arguments = self.__arguments._replace(
            LEARNING_RATE=best.hyperparameters.get('learning_rate'),
            WEIGHT_DECAY=best.hyperparameters.get('weight_decay'))

        # Then
        src.models.prime.Prime(
            enumerator=self.__enumerator, archetype=self.__archetype,
            arguments=self.__arguments, tokenizer=self.__tokenizer).exc(
            training=training, validating=validating)

