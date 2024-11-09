"""Module measurements.py"""
import logging
import os.path

import seqeval.metrics as sme
import sklearn.metrics as sm

import config
import src.models.numerics
import src.elements.arguments as ag
import src.functions.objects
import src.functions.directories


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
        self.__arguments = arguments

        self.__configurations = config.Config()
        self.__objects = src.functions.objects.Objects()

        # Logging
        logging.basicConfig(level=logging.INFO, format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def __sci(self):
        """

        :return:
        """

        report = sm.classification_report(y_true=self.__originals, y_pred=self.__predictions, zero_division=0.0)
        self.__logger.info('scikit-learn:\n%s', report)

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

        self.__logger.info('SEQ:\n%s\n%s', report, accuracy)

        return report, accuracy

    def __numerics(self) -> dict:
        """

        :return:
        """

        values: dict = src.models.numerics.Numerics(
            originals=self.__originals, predictions=self.__predictions).exc()
        self.__logger.info('numerics:\n%s', values)

        return values

    def exc(self, segment: str):
        """

        :param segment: prime or hyperparameters
        :return:
        """

        path = os.path.join(self.__configurations.artefacts_, self.__arguments.architecture, segment, 'metrics')
        src.functions.directories.Directories().create(path=path)

        fine = self.__sci()
        self.__objects.write(nodes=fine, path=os.path.join(path, 'fine.json'))

        coarse, _ = self.__seq()
        self.__objects.write(nodes=coarse, path=os.path.join(path, 'coarse.json'))

        fundamental = self.__numerics()
        self.__objects.write(nodes=fundamental, path=os.path.join(path, 'fundamental.json'))
