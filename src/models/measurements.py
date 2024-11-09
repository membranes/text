"""Module measurements.py"""
import logging

import seqeval.metrics as sme
import sklearn.metrics as sm
import src.models.numerics
import src.elements.arguments as ag


class Measurements:
    """
    For classification metrics calculations
    """

    def __init__(self, originals: list, predictions: list, arguments: ag.Arguments):
        """

        :param originals: The true values, a simple, i.e., un-nested, list.
        :param predictions: The predictions, a simple list, i.e., un-nested, list.
        """

        self.__originals = originals
        self.__predictions = predictions

        # Logging
        logging.basicConfig(level=logging.INFO, format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def __sci(self):
        """

        :return:
        """

        report = sm.classification_report(y_true=self.__originals, y_pred=self.__predictions, zero_division=0.0)

        return report

    def __seq(self):
        """

        :return:
        """

        y_true = [self.__originals]
        y_pred = [self.__predictions]

        # str
        report = sme.classification_report(y_true=y_true, y_pred=y_pred, zero_division=0.0)

        # float
        accuracy: float = sme.accuracy_score(y_true=y_true, y_pred=y_pred)

        return report, accuracy

    def __numerics(self) -> dict:
        """

        :return:
        """

        values: dict = src.models.numerics.Numerics(
            originals=self.__originals, predictions=self.__predictions).exc()

        return values

    def exc(self):
        """

        :return:
        """

        fine = self.__sci()
        coarse = self.__seq()
        fundamental = self.__numerics()
