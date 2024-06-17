"""Module intelligence.py"""
import torch.utils.data as tu
import transformers

import src.elements.structures as sr
import src.elements.variable as vr
import src.models.distil.parameters


class Intelligence:
    """
    Intelligence
    """

    def __init__(self, variable: vr.Variable, enumerator: dict):
        """

        :param variable:
        :param enumerator:
        """

        self.__variable = variable

        # Parameters
        self.__parameters = src.models.distil.parameters.Parameters()

        # https://huggingface.co/docs/transformers/v4.41.3/en/model_doc/auto#transformers.AutoModel
        self.__model = transformers.AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=self.__parameters.model_checkpoint,
            **{'num_labels': len(enumerator)})

        # https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/trainer#transformers.TrainingArguments
        self.__args = transformers.TrainingArguments(
            output_dir=self.__parameters.model_checkpoint.split('/')[-1],
            evaluation_strategy='epoch',
            learning_rate=self.__variable.LEARNING_RATE,
            per_device_train_batch_size=self.__variable.TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=self.__variable.VALID_BATCH_SIZE,
            num_train_epochs=self.__variable.EPOCHS,
            weight_decay=0.01)

    def __call__(self, training: sr.Structures, validating: sr.Structures):
        """
        https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/trainer#transformers.Trainer

        :return:
        """

        transformers.Trainer(model=self.__model, args=self.__args, train_dataset=training.dataset, eval_dataset=validating.dataset)
