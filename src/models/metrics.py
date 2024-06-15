import logging

import seqeval.metrics as sme


class Metrics:

    def __init__(self):
        # Logging
        logging.basicConfig(level=logging.INFO, format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def exc(self, originals: list, predictions: list):
        """

        :param originals: The true values
        :param predictions: The predictions
        :return:
        """

        self.__logger.info(
            sme.classification_report(y_true=originals, y_pred=predictions))

        self.__logger.info(
            sme.accuracy_score(y_true=originals, y_pred=predictions))
