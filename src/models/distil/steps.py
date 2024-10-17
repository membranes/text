"""Module steps.py"""
import logging

import transformers.tokenization_utils_base

import src.elements.arguments as ag
import src.elements.vault as vu
import src.models.distil.structures
import src.models.distil.tokenizer
import src.models.measurements
import src.models.operating
import src.models.optimal
import src.models.validation


class Steps:
    """
    Steps
    ref. https://huggingface.co/docs/transformers/tasks/token_classification
    """

    def __init__(self, enumerator: dict, archetype: dict, arguments: ag.Arguments, vault: vu.Vault):
        """

        :param enumerator: The tags and their identification codes.
        :param archetype: The inverse dict of enumerator.
        :param arguments: The parameter values for ...
        :param vault: An object of dataframes, consisting of the training, validating, and testing data sets.
        """

        # Inputs
        self.__enumerator = enumerator
        self.__archetype = archetype
        self.__arguments = arguments
        self.__vault = vault

        # A set of values for machine learning model development
        self.__arguments = self.__arguments._replace(
            N_TRAIN=self.__vault.training.shape[0], N_VALID=self.__vault.validating.shape[0], N_TEST=self.__vault.testing.shape[0])

        # Get tokenizer
        self.__tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase = (
            src.models.distil.tokenizer.Tokenizer(arguments=self.__arguments)())

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def __structures(self):
        """

        :return:
        """

        structures = src.models.distil.structures.Structures(
            enumerator=self.__enumerator, arguments=self.__arguments,
            vault=self.__vault, tokenizer=self.__tokenizer)

        return structures.training(), structures.validating(), structures.testing()

    def exc(self):
        """

        :return:
        """

        training, validating, _ = self.__structures()
        self.__logger.info(self.__arguments)

        # Hyperparameter search
        optimal = src.models.optimal.Optimal(
            arguments=self.__arguments, enumerator=self.__enumerator, archetype=self.__archetype)
        best = optimal(training=training, validating=validating, tokenizer=self.__tokenizer)
        self.__logger.info(best)

        # Hence, update the modelling variables
        self.__arguments = self.__arguments._replace(
            LEARNING_RATE=best.hyperparameters.get('learning_rate'), WEIGHT_DECAY=best.hyperparameters.get('weight_decay'),
            TRAIN_BATCH_SIZE=best.hyperparameters.get('per_device_train_batch_size'))
        self.__logger.info(self.__arguments)

        # Training via the best hyperparameters set
        operating = src.models.operating.Operating(
            arguments=self.__arguments, enumerator=self.__enumerator, archetype=self.__archetype)
        model = operating.exc(training=training, validating=validating, tokenizer=self.__tokenizer)
        self.__logger.info(dir(model))

        # Evaluating: vis-à-vis model & validation data
        originals, predictions = src.models.validation.Validation(
            validating=validating, archetype=self.__archetype).exc(model=model)

        src.models.measurements.Measurements().exc(
            originals=originals, predictions=predictions)
