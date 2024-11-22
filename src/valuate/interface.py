"""Module interface.py"""
import datasets
import transformers

import src.valuate.estimates
import src.valuate.measurements


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

    def exc(self, blob: datasets.Dataset, path: str):
        """

        :param blob:
        :param path:
        :return:
        """

        originals, predictions = src.valuate.estimates.Estimates(
            blob=blob, archetype=self.__archetype).exc(model=self.__model)

        src.valuate.measurements.Measurements(
            originals=originals, predictions=predictions).exc(path=path)
