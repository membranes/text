"""Module steps.py"""
import logging
import os.path

import transformers.tokenization_utils_base

import src.elements.arguments as ag
import src.elements.hyperspace as hp
import src.elements.vault as vu
import src.models.distil.structures
import src.models.distil.tokenizer
import src.models.hyperpoints
import src.models.prime


class Steps:
    """
    Steps
    ref. https://huggingface.co/docs/transformers/tasks/token_classification
    """

    def __init__(self, enumerator: dict, archetype: dict, arguments: ag.Arguments, hyperspace: hp.Hyperspace, vault: vu.Vault):
        """

        :param enumerator: Of tags; key &rarr; identifier, value &rarr; label<br>
        :param archetype: Of tags; key &rarr; label, value &rarr; identifier<br>
        :param arguments: A suite of values/arguments for machine learning model development.<br>
        :param hyperspace: The hyperparameters alongside their starting values or number spaces.<br>
        :param vault: An object of dataframes, consisting of the training, validating, and testing data sets.<br>
        """

        # Inputs
        self.__enumerator = enumerator
        self.__archetype = archetype
        self.__arguments = arguments
        self.__hyperspace = hyperspace
        self.__vault = vault

        # A set of values for machine learning model development
        self.__arguments = self.__arguments._replace(
            N_TRAIN=self.__vault.training.shape[0], N_VALID=self.__vault.validating.shape[0],
            N_TEST=self.__vault.testing.shape[0])

        # Get tokenizer
        self.__tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase = (
            src.models.distil.tokenizer.Tokenizer(arguments=self.__arguments)())

    def exc(self):
        """

        :return:
        """

        structures = src.models.distil.structures.Structures(
            enumerator=self.__enumerator, arguments=self.__arguments,
            vault=self.__vault, tokenizer=self.__tokenizer)
        training = structures.training()
        validating = structures.validating()

        # Hyperparameter search
        section = os.path.join(self.__arguments.model_output_directory, 'hyperparameters')
        self.__arguments = self.__arguments._replace(model_output_directory=section)
        optimal = src.models.hyperpoints.Hyperpoints(
            arguments=self.__arguments, hyperspace=self.__hyperspace,
            enumerator=self.__enumerator, archetype=self.__archetype)
        best = optimal(training=training, validating=validating, tokenizer=self.__tokenizer)
        logging.info(best)

        # Hence, update the modelling variables
        self.__arguments = self.__arguments._replace(
            LEARNING_RATE=best.hyperparameters.get('learning_rate'),
            WEIGHT_DECAY=best.hyperparameters.get('weight_decay'),
            TRAIN_BATCH_SIZE=best.hyperparameters.get('per_device_train_batch_size'))

        # Then
        section = os.path.join(os.path.dirname(section), 'prime')
        self.__arguments = self.__arguments._replace(model_output_directory=section)
        src.models.prime.Prime(
            enumerator=self.__enumerator, archetype=self.__archetype,
            arguments=self.__arguments, tokenizer=self.__tokenizer).exc(
            training=training, validating=validating)
