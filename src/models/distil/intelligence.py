"""Module intelligence.py"""
import transformers

import src.models.distil.parameters as pr


class Intelligence:
    """
    Intelligence
    """

    def __init__(self, enumerator: dict, archetype: dict):
        """

        :param enumerator: key -> identifier, value -> label
        :param archetype: key -> label, value -> identifier
        """

        self.__enumerator = enumerator
        self.__archetype = archetype

        # Parameters
        self.__parameters = pr.Parameters()

    @staticmethod
    def collator(tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase) -> transformers.DataCollatorForTokenClassification:
        """

        :param tokenizer:
        :return:
        """

        return transformers.DataCollatorForTokenClassification(tokenizer=tokenizer)

    def model(self):
        """
        https://huggingface.co/docs/transformers/v4.41.3/en/model_doc/auto#transformers.AutoModel

        transformers.AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=self.__parameters.pretrained_model_name,
            config=config
        )

        :return:
        """

        config = transformers.DistilBertConfig(dropout=0.1, activation='gelu').from_pretrained(
            pretrained_model_name_or_path=self.__parameters.pretrained_model_name,
            **{'num_labels': len(self.__enumerator)})

        return transformers.DistilBertForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=self.__parameters.pretrained_model_name,
            config=config
        )
