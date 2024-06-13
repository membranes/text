
import transformers

import torch

import src.models.bert.parameters


class Modelling:

    def __init__(self, enumerator: dict):
        """

        :param enumerator: The labels enumerator
        """

        # The device for computation
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained
        # https://huggingface.co/docs/transformers/main_classes/configuration#transformers.PretrainedConfig
        self.__parameters = src.models.bert.parameters.Parameters()

        self.model = transformers.BertForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=self.__parameters.pretrained_model_name,
            **{'num_labels': len(enumerator)})

        self.model.to(device)
