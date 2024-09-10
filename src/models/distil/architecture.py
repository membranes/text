import logging
import os

import transformers

import src.elements.structures as sr
import src.elements.variable as vr
import src.models.distil.intelligence
import src.models.distil.parameters as pr
import src.models.distil.storage
import src.models.distil.metrics
import config


class Architecture:

    def __init__(self, variable: vr.Variable, enumerator: dict):
        """

        :param variable:
        :param enumerator:
        """

        self.__variable = variable
        self.__enumerator = enumerator

        # Parameters
        self.__parameters = pr.Parameters()

        # Directory preparation
        src.models.distil.storage.Storage().exc(path=self.__parameters.path)



    def __args(self):

        # https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/trainer#transformers.TrainingArguments
        return transformers.TrainingArguments(
            output_dir=self.__parameters.path,
            eval_strategy='epoch',
            save_strategy='epoch',
            learning_rate=self.__variable.LEARNING_RATE,
            weight_decay=self.__variable.WEIGHT_DECAY,
            per_device_train_batch_size=self.__variable.TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=self.__variable.VALID_BATCH_SIZE,
            num_train_epochs=self.__variable.EPOCHS,
            max_steps=-1,
            warmup_steps=0,
            logging_dir=os.path.join(self.__parameters.path, 'tensorboard'),
            no_cuda=False,
            seed=config.Config().seed,
            save_total_limit=5,
            skip_memory_metrics=True,
            load_best_model_at_end=True,
            fp16=True,
            push_to_hub=False
            )


    def __call__(self, training: sr.Structures, validating: sr.Structures,
                 tokenizer: transformers.PreTrainedTokenizerBase):
        """
        https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/trainer#transformers.Trainer

        :return:
        """

        # Intelligence
        intelligence = src.models.distil.intelligence.Intelligence(enumerator=self.__enumerator)
        model = intelligence.model()
        data_collator = intelligence.collator(tokenizer=tokenizer)

        metrics = src.models.distil.metrics.Metrics()


        # Hence
        model.to(self.__parameters.device)

        trainer = transformers.Trainer(
            model=model,
            args=self.__args(),
            data_collator=data_collator,
            train_dataset=training.dataset,
            eval_dataset=validating.dataset,
            tokenizer=tokenizer,
            compute_metrics=metrics.exc)

        best = trainer.hyperparameter_search(
            n_trials=2
        )

        return best
