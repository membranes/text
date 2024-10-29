"""Module measurements.py"""
import logging

import seqeval.metrics as sme
import sklearn.metrics as sm


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

        self.__logger.info('SCIKIT LEARN')

        self.__logger.info(
            sm.classification_report(y_true=originals, y_pred=predictions, zero_division=0.0))

    def __seq(self, originals: list, predictions: list):
        """

        :param originals: The true values
        :param predictions: The predictions
        :return:
        """

        y_true = [originals]
        y_pred = [predictions]

        report = sme.classification_report(y_true=y_true, y_pred=y_pred, zero_division=0.0)
        self.__logger.info('SEQ EVAL\n%s\n%s', type(report), report)

        accuracy = sme.accuracy_score(y_true=y_true, y_pred=y_pred)
        self.__logger.info('\n%s\n%s', type(accuracy), accuracy)

    def exc(self, originals: list, predictions: list):
        """

        :param originals: The true values
        :param predictions: The predictions
        :return:
        """

        self.__logger.info('In Measurements:\n%s', originals)
        self.__logger.info('In Measurements:\n%s', predictions)

        self.__sci(originals=originals, predictions=predictions)
        self.__seq(originals=originals, predictions=predictions)
