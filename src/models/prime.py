import transformers

import src.elements.arguments as ag
import src.models.measurements
import src.models.operating
import src.models.validation


class Prime:

    def __init__(self, enumerator: dict, archetype: dict, arguments: ag.Arguments,
                 tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase):
        """

        :param enumerator:
        :param archetype:
        :param arguments:
        :param tokenizer:
        """

        self.__enumerator = enumerator
        self.__archetype = archetype
        self.__arguments = arguments
        self.__tokenizer = tokenizer

    def exc(self, training, validating):
        """

        :param training:
        :param validating:
        :return:
        """

        # Training via the best hyperparameters set
        operating = src.models.operating.Operating(
            arguments=self.__arguments, enumerator=self.__enumerator, archetype=self.__archetype)
        model = operating.exc(training=training, validating=validating, tokenizer=self.__tokenizer)

        # Evaluating: vis-à-vis model & validation data
        originals, predictions = src.models.validation.Validation(
            validating=validating, archetype=self.__archetype).exc(model=model)

        src.models.measurements.Measurements().exc(
            originals=originals, predictions=predictions)

