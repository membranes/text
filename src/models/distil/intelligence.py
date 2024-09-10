"""Module intelligence.py"""
import logging

import transformers

import src.elements.structures as sr
import src.elements.variable as vr
import src.models.distil.parameters as pr
import src.models.distil.storage


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
        self.__parameters = pr.Parameters()

        # Directory preparation
        src.models.distil.storage.Storage().exc(path=self.__parameters.path)

        # https://huggingface.co/docs/transformers/v4.41.3/en/model_doc/auto#transformers.AutoModel
        self.__model = transformers.AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=self.__parameters.pretrained_model_name,
            **{'num_labels': len(enumerator)})
        self.__model.to(self.__parameters.device)

    def __args(self):

        # https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/trainer#transformers.TrainingArguments
        return transformers.TrainingArguments(
            output_dir=self.__parameters.path,
            eval_strategy='epoch',
            learning_rate=self.__variable.LEARNING_RATE,
            per_device_train_batch_size=self.__variable.TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=self.__variable.VALID_BATCH_SIZE,
            num_train_epochs=self.__variable.EPOCHS,
            weight_decay=0.01)


    def __call__(self, training: sr.Structures, validating: sr.Structures,
                 tokenizer: transformers.PreTrainedTokenizerBase) -> transformers.Trainer:
        """
        https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/trainer#transformers.Trainer

        :return:
        """

        trainer = transformers.Trainer(
            model=self.__model, args=self.__args(), train_dataset=training.dataset,
            eval_dataset=validating.dataset, tokenizer=tokenizer)

        trainer.train()

        return trainer
