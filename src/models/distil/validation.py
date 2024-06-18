import logging

import transformers
import numpy as np

import src.elements.structures as sr


class Validation:
    """Validation Class

    <br>
    Summarises a variety of validation steps

    """

    def __init__(self, validating: sr.Structures):
        """

        :param validating:
        """

        self.__validating = validating

        # Logging
        logging.basicConfig(level=logging.INFO, format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H-%M-%S')
        self.__logger = logging.getLogger(__name__)

    def exc(self, model: transformers.Trainer):
        """

        :param model:
        :return:
        """

        bucket = model.predict(self.__validating.dataset)

        self.__logger.info('Determining active labels & predictions')
        __labels: np.ndarray = bucket.label_ids
        __predictions: np.ndarray = bucket.predictions

        self.__logger.info('Labels: %s', __labels.shape)
        self.__logger.info('Predictions: %s', __predictions.shape)

        l = __labels.reshape(-1)
        p = __predictions.reshape(-1, model.model.config.num_labels)

        self.__logger.info('Labels: %s', l.shape)
        self.__logger.info('Predictions: %s', p.shape)

        '''
        active = np.not_equal(l, -100)
        labels = l[active]
        predictions = p[active]

        self.__logger.info(labels.shape)
        self.__logger.info(predictions.shape)

        self.__logger.info(labels)
        self.__logger.info(predictions)
        '''

        self.__logger.info(bucket.__doc__)
