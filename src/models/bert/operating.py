import transformers.trainer_utils

import src.elements.arguments as ag
import src.elements.structures as sr
import src.models.args
import src.models.metrics
import src.models.algorithm


class Operating:

    def __init__(self, arguments: ag.Arguments, enumerator: dict, archetype: dict):
        """

        :param arguments:
        :param enumerator:
        :param archetype:
        """

        self.__arguments = arguments
        self.__enumerator = enumerator
        self.__archetype = archetype

        # Intelligence
        self.__algorithm = src.models.algorithm.Algorithm(architecture=self.__arguments.name)

    def exc(self, training: sr.Structures, validating: sr.Structures,
            tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase):
        """

        :param training:
        :param validating:
        :param tokenizer:
        :return:
        """

        # Arguments
        args = src.models.args.Args(arguments=self.__arguments).exc()

        # Model
        intelligence = self.__algorithm.exc(
            arguments=self.__arguments, enumerator=self.__enumerator, archetype=self.__archetype)

        # Metrics
        metrics = src.models.metrics.Metrics(archetype=self.__archetype)

        # Temporary
        data_collator: transformers.DataCollatorForTokenClassification = (
            transformers.DataCollatorForTokenClassification(tokenizer=tokenizer))

        # Trainer
        trainer = transformers.Trainer(
            model_init=intelligence.model,
            args=args, data_collator=data_collator,
            train_dataset=training.dataset, eval_dataset=validating.dataset,
            tokenizer=tokenizer,
            compute_metrics=metrics.exc)

        trainer.train()

        return trainer
