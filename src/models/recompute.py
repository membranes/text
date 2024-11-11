"""Training via best set of hyperparameters."""
import datasets
import transformers.trainer_utils

import src.elements.arguments as ag
import src.models.algorithm
import src.models.metrics
import src.models.training_arguments


class Recompute:
    """
    Class Recompute
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

    def exc(self, training: datasets.Dataset, validating: datasets.Dataset,
            tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase):
        """

        :param training: The training data object.<br>
        :param validating: The validation data object.<br>
        :param tokenizer: The tokenizer of text.<br>
        :return:
        """

        # Training Arguments: Only save the checkpoint at the optimal training point.
        self.__arguments = self.__arguments._replace(
            EPOCHS=2*self.__arguments.EPOCHS, save_total_limit=1)
        args = src.models.training_arguments.TrainingArguments(arguments=self.__arguments).exc()

        # Model
        algorithm = self.__algorithm.exc(
            arguments=self.__arguments, enumerator=self.__enumerator, archetype=self.__archetype)

        # Metrics
        metrics = src.models.metrics.Metrics(archetype=self.__archetype)

        # Trainer
        trainer = transformers.Trainer(
            model_init=algorithm.model,
            args=args,
            data_collator=transformers.DataCollatorForTokenClassification(tokenizer=tokenizer),
            train_dataset=training,
            eval_dataset=validating,
            tokenizer=tokenizer,
            compute_metrics=metrics.exc)

        trainer.train()

        return trainer
