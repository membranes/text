import transformers.trainer_utils

import src.elements.arguments as ag
import src.elements.structures as sr
import src.models.args
import src.models.bert.intelligence
import src.models.metrics


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

        # Intelligence
        intelligence = src.models.bert.intelligence.Intelligence(
            enumerator=self.__enumerator, archetype=self.__archetype, arguments=self.__arguments)

        # Metrics
        metrics = src.models.metrics.Metrics(archetype=self.__archetype)

        # Trainer
        trainer = transformers.Trainer(
            model_init=intelligence.model,
            args=args, data_collator=intelligence.collator(tokenizer),
            train_dataset=training.dataset, eval_dataset=validating.dataset,
            tokenizer=tokenizer,
            compute_metrics=metrics.exc)

        trainer.train()

        return trainer
