"""Module interface.py"""
import datasets
import transformers

import src.elements.arguments as ag
import src.evaluate.estimates
import src.evaluate.measurements


class Interface:
    """
    Interface
    """

    def __init__(self, model: transformers.Trainer, archetype: dict):
        """

        :param model:
        :param archetype:
        """

        self.__model = model
        self.__archetype = archetype

    def exc(self, blob: datasets.Dataset, arguments: ag.Arguments, path: str):
        """

        :param blob:
        :param arguments:
        :param path:
        :return:
        """

        originals, predictions = src.evaluate.estimates.Estimates(
            blob=blob, archetype=self.__archetype).exc(model=self.__model)

        src.evaluate.measurements.Measurements(
            originals=originals, predictions=predictions, arguments=arguments).exc(path=path)

