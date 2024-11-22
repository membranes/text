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

    def __init__(self, model: transformers.Trainer, archetype: dict, arguments: ag.Arguments,):
        """

        :param model:
        :param archetype:
        :param arguments:
        """

        self.__model = model
        self.__archetype = archetype
        self.__arguments = arguments

    def exc(self, blob: datasets.Dataset, path: str):
        """

        :param blob:
        :param arguments:
        :param path:
        :return:
        """

        originals, predictions = src.evaluate.estimates.Estimates(
            blob=blob, archetype=self.__archetype).exc(model=self.__model)

        src.evaluate.measurements.Measurements(
            originals=originals, predictions=predictions, arguments=self.__arguments).exc(path=path)

