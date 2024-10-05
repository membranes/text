"""Module arguments.py"""
import os
import transformers

import src.elements.variable as vr

import src.models.distil.parameters

import config


class Arguments:
    """
    Class Arguments
    """

    def __init__(self, variable: vr.Variable):
        """

        :param variable:
        """

        self.__variable = variable

        self.__parameters = src.models.distil.parameters.Parameters()

    def exc(self):
        """
        https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/trainer#transformers.TrainingArguments

        TensorBoard logging directory: output_dir/runs/CURRENT_DATETIME_HOSTNAME*
            https://huggingface.co/docs/transformers/v4.44.2/en/main_classes/
                &amp;num;transformers.TrainingArguments.logging_dir

        :return:
        """

        return transformers.TrainingArguments(
            output_dir=self.__parameters.MODEL_OUTPUT_DIRECTORY,
            eval_strategy='epoch',
            save_strategy='epoch',
            learning_rate=self.__variable.LEARNING_RATE,
            weight_decay=self.__variable.WEIGHT_DECAY,
            per_device_train_batch_size=self.__variable.TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=self.__variable.VALID_BATCH_SIZE,
            num_train_epochs=self.__variable.EPOCHS,
            max_steps=-1,
            warmup_steps=0,
            no_cuda=False,
            seed=config.Config().seed,
            save_total_limit=5,
            skip_memory_metrics=True,
            load_best_model_at_end=True,
            logging_dir=os.path.join(self.__parameters.MODEL_OUTPUT_DIRECTORY, 'logs'),
            fp16=True,
            push_to_hub=False)