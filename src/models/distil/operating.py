import transformers.trainer_utils

import src.elements.structures as sr
import src.elements.variable as vr
import src.models.distil.arguments
import src.models.distil.intelligence
import src.models.distil.metrics


class Operating:
    """
    Class Operating
    """

    def __init__(self, variable: vr.Variable, enumerator: dict, archetype: dict):
        """

        :param variable:
        :param enumerator:
        :param archetype:
        """

        self.__variable = variable
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
        args = src.models.distil.arguments.Arguments(variable=self.__variable).exc()

        # Intelligence
        intelligence = src.models.distil.intelligence.Intelligence(
            enumerator=self.__enumerator, archetype=self.__archetype)

        # Metrics
        metrics = src.models.distil.metrics.Metrics(archetype=self.__archetype)

        # Trainer
        trainer = transformers.Trainer(
            model_init=intelligence.model,
            args=args, data_collator=intelligence.collator(tokenizer),
            train_dataset=training.dataset, eval_dataset=validating.dataset,
            tokenizer=tokenizer,
            compute_metrics=metrics.exc)

        trainer.train()

        return trainer