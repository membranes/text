"""Module measurements.py"""
import logging

import seqeval.metrics as sme
import sklearn.metrics as sm
import src.models.numerics


class Measurements:
    """
    For classification metrics calculations
    """

    def __init__(self):
        """
        Constructor
        """

        # Logging
        logging.basicConfig(level=logging.INFO, format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def __sci(self, originals: list, predictions: list):
        """

        :param originals: The true values
        :param predictions: The predictions
        :return:
        """

        report = sm.classification_report(y_true=originals, y_pred=predictions, zero_division=0.0)
        self.__logger.info('SCIKIT LEARN:\n%s\n%s', type(report), report)

    def __seq(self, originals: list, predictions: list):
        """

        :param originals: The true values
        :param predictions: The predictions
        :return:
        """

        y_true = [originals]
        y_pred = [predictions]

        report = sme.classification_report(y_true=y_true, y_pred=y_pred, zero_division=0.0)
        self.__logger.info('\n\nSEQ EVAL:\n%s\n%s', type(report), report)

        accuracy = sme.accuracy_score(y_true=y_true, y_pred=y_pred)
        self.__logger.info('\n%s\n%s', type(accuracy), accuracy)

    def __numerics(self, originals: list, predictions: list):

        values = src.models.numerics.Numerics(
            originals=originals, predictions=predictions).exc()

        self.__logger.info('NUMERICS:\n%s', values)

    def exc(self, originals: list, predictions: list):
        """

        :param originals: The true values, simple list, i.e., not nested.
        :param predictions: The predictions, simple list, i.e., not nested.
        :return:
        """

        self.__sci(originals=originals, predictions=predictions)
        self.__seq(originals=originals, predictions=predictions)
        self.__numerics(originals=originals, predictions=predictions)
