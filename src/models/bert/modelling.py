
import transformers

import src.models.bert.parameters


class Modelling:

    def __init__(self, device: str, enumerator: dict):
        """

        :param device: The machine device for computation
        :param enumerator: The labels enumerator
        """

        self.__parameters = src.models.bert.parameters.Parameters()

        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained
        # https://huggingface.co/docs/transformers/main_classes/configuration#transformers.PretrainedConfig
        self.model = transformers.BertForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=self.__parameters.pretrained_model_name,
            **{'num_labels': len(enumerator)})
        self.model.to(device)
