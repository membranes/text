"""Module intelligence.py"""
import torch.utils.data as tu
import transformers

import src.elements.variable
import src.models.distil.parameters
import src.models.distil.parameters


class Intelligence:
    """
    Intelligence
    """

    def __init__(self, variable: src.elements.variable.Variable,
                 enumerator: dict, dataloader: tu.DataLoader):
        """

        :param variable:
        :param enumerator:
        :param dataloader:
        """

        self.__variable = variable
        self.__dataloader = dataloader

        # Parameters
        self.__parameters = src.models.distil.parameters.Parameters()

        # https://huggingface.co/docs/transformers/v4.41.3/en/model_doc/auto#transformers.AutoModel
        model = transformers.AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=self.__parameters.model_checkpoint,
            **{'num_labels': len(enumerator)})

        # https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/trainer#transformers.TrainingArguments
        transformers.TrainingArguments(
            output_dir=self.__parameters.model_checkpoint.split('/')[-1],
            evaluation_strategy='epoch',
            learning_rate=self.__variable.LEARNING_RATE,
            per_device_train_batch_size=self.__variable.TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=self.__variable.VALID_BATCH_SIZE,
            num_train_epochs=self.__variable.EPOCHS,
            weight_decay=0.01)

    def __call__(self):
        """
        https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/trainer#transformers.Trainer

        :return:
        """
