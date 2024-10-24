"""Module prime.py"""
import transformers

import src.elements.arguments as ag
import src.models.measurements
import src.models.recompute
import src.models.validation


class Prime:
    """
    Notes<br>
    ------<br>

    Determines the prime model vis-à-vis the best set of hyperparameters.
    """

    def __init__(self, enumerator: dict, archetype: dict, arguments: ag.Arguments,
                 tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase):
        """

        :param enumerator: Of tags; key &rarr; identifier, value &rarr; label<br>
        :param archetype: Of tags; key &rarr; label, value &rarr; identifier<br>
        :param arguments: A suite of values/arguments for machine learning model development.<br>
        :param tokenizer: The tokenizer of text.<br>
        """

        self.__enumerator = enumerator
        self.__archetype = archetype
        self.__arguments = arguments
        self.__tokenizer = tokenizer

    def exc(self, training, validating):
        """

        :param training: The training data.
        :param validating: The validation data.
        :return:
        """

        # Training via the best hyperparameters set
        operating = src.models.recompute.Recompute(
            arguments=self.__arguments, enumerator=self.__enumerator, archetype=self.__archetype)
        model = operating.exc(training=training, validating=validating, tokenizer=self.__tokenizer)

        # Evaluating: vis-à-vis model & validation data
        originals, predictions = src.models.validation.Validation(
            validating=validating, archetype=self.__archetype).exc(model=model)

        src.models.measurements.Measurements().exc(
            originals=originals, predictions=predictions)
