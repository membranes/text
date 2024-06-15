"""Module parameters.py"""
import torch
import transformers


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
        self.model_checkpoint = 'distilbert_base_uncased'

        # The device for computation
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

