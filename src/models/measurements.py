"""Module measurements.py"""
import logging

import seqeval.metrics as sme
import sklearn.metrics as sm
import src.models.numerics


class Measurements:
    """
    For classification metrics calculations
    """

    def __init__(self, originals: list, predictions: list):
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
        self.__logger.info('SCIKIT LEARN:\n%s\n%s', type(report), report)

    def __seq(self):
        """

        :return:
        """

        y_true = [self.__originals]
        y_pred = [self.__predictions]

        report = sme.classification_report(y_true=y_true, y_pred=y_pred, zero_division=0.0)
        self.__logger.info('\n\nSEQ EVAL:\n%s\n%s', type(report), report)

        accuracy = sme.accuracy_score(y_true=y_true, y_pred=y_pred)
        self.__logger.info('\n%s\n%s', type(accuracy), accuracy)

    def __numerics(self):
        """

        :return:
        """

        values = src.models.numerics.Numerics(
            originals=self.__originals, predictions=self.__predictions).exc()

        self.__logger.info('\n\nNUMERICS:\n%s', values)

    def exc(self):
        """

        :return:
        """

        self.__sci()
        self.__seq()
        self.__numerics()
