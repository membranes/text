"""Module tokenizer.py"""
import transformers

import src.models.bert.parameters


class Tokenizer:

    def __init__(self):
        """
        Constructor
        """

        self.__parameters = src.models.bert.parameters.Parameters()

    def __call__(self) -> transformers.tokenization_utils_base.PreTrainedTokenizerBase:
        """

        :return:
        """

        # Tokenizer
        return transformers.BertTokenizerFast.from_pretrained(
            pretrained_model_name_or_path=self.__parameters.pretrained_model_name,
            clean_up_tokenization_spaces=True)
