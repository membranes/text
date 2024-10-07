"""Module architecture.py"""

import transformers

import src.elements.structures as sr
import src.elements.variable as vr
import src.functions.directories
import src.models.bert.parameters as pr
import src.models.bert.arguments
import src.models.bert.intelligence
import src.models.bert.metrics
import src.models.bert.settings


class Architecture:
    """
    Architecture
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

        # Parameters
        self.__parameters = pr.Parameters()

        # Directory preparation
        src.functions.directories.Directories().cleanup(path=self.__parameters.storage_path)

    def __call__(self, training: sr.Structures, validating: sr.Structures,
                 tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase):
        """
        https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/trainer#transformers.Trainer
        https://docs.ray.io/en/latest/tune/api/doc/ray.tune.run.html

        :param training:
        :param validating:
        :param tokenizer:
        :return:
        """

        # Arguments
        args = src.models.bert.arguments.Arguments(variable=self.__variable).exc()

        # Collator, Model, ETC.
        intelligence = src.models.bert.intelligence.Intelligence(enumerator=self.__enumerator, archetype=self.__archetype)

        # Metrics
        metrics = src.models.bert.metrics.Metrics(archetype=self.__archetype)

        # Settings
        settings = src.models.bert.settings.Settings(variable=self.__variable)
