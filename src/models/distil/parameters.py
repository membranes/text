"""Module parameters.py"""
import os
import torch
import transformers

import src.functions.directories


class Parameters:
    """

    [from_pretrained](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained)
    [PretrainedConfig](https://huggingface.co/docs/transformers/main_classes/configuration#transformers.PretrainedConfig)
    """

    def __init__(self):
        """
        Constructor
        """

        self.task = 'ner'

        # The device for computation
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Pretrained model
        self.pretrained_model_name = 'distilbert-base-uncased'

        # Directories
        path = os.path.join(os.getcwd(), self.pretrained_model_name)
        directories = src.functions.directories.Directories()
        directories.cleanup(path=path)

        # Tokenizer
        self.tokenizer = transformers.DistilBertTokenizerFast.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_model_name)
