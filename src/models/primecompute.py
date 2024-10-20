"""Module prime_steps.py"""
import transformers.trainer_utils

import src.elements.arguments as ag
import src.elements.structures as sr
import src.models.algorithm
import src.models.trainee
import src.models.metrics


class PrimeSteps:
    """
    Class Operating
    """

    def __init__(self, arguments: ag.Arguments, enumerator: dict, archetype: dict):
        """

        :param arguments: A suite of values/arguments for machine learning model development.<br>
        :param enumerator: Of tags; key &rarr; identifier, value &rarr; label<br>
        :param archetype: Of tags; key &rarr; label, value &rarr; identifier<br>
        """

        self.__arguments = arguments
        self.__enumerator = enumerator
        self.__archetype = archetype

        # Intelligence
        self.__algorithm = src.models.algorithm.Algorithm(architecture=self.__arguments.architecture)

    def exc(self, training: sr.Structures, validating: sr.Structures,
            tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase):
        """

        :param training: The training data object.<br>
        :param validating: The validation data object.<br>
        :param tokenizer: The tokenizer of text.<br>
        :return:
        """

        # Training Arguments
        args = src.models.trainee.Trainee(arguments=self.__arguments).exc()

        # Model
        intelligence = self.__algorithm.exc(
            arguments=self.__arguments, enumerator=self.__enumerator, archetype=self.__archetype)

        # Metrics
        metrics = src.models.metrics.Metrics(archetype=self.__archetype)

        # Trainer
        trainer = transformers.Trainer(
            model_init=intelligence.model,
            args=args,
            data_collator=transformers.DataCollatorForTokenClassification(tokenizer=tokenizer),
            train_dataset=training.dataset,
            eval_dataset=validating.dataset,
            tokenizer=tokenizer,
            compute_metrics=metrics.exc)

        trainer.train()

        return trainer
