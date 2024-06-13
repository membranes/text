
import transformers

import src.structures.bert.parameters


class Modelling:

    def __init__(self, device: str, enumerator: dict):
        """

        :param device: The machine device for computation
        :param enumerator: The labels enumerator
        """

        self.__parameters = src.structures.bert.parameters.Parameters()

        # https://huggingface.co/docs/transformers/main_classes/configuration#transformers.PretrainedConfig
        model = transformers.BertForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=self.__parameters.pretrained_model_name,
            **{'num_labels': len(enumerator)})
        model.to(device)
