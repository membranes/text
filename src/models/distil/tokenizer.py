"""Module tokenizer"""
import transformers

import src.elements.arguments as ag


class Tokenizer:

    def __init__(self, arguments: ag.Arguments):
        """
        Constructor
        """

        self.__arguments = arguments

    def __call__(self) -> transformers.tokenization_utils_base.PreTrainedTokenizerBase:
        """

        :return:
        """

        # Tokenizer
        return transformers.DistilBertTokenizerFast.from_pretrained(
            pretrained_model_name_or_path=self.__arguments.pretrained_model_name,
            clean_up_tokenization_spaces=True)
