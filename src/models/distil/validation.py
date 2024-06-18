import logging

import transformers

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

        self.__logger.info(bucket.metrics)
        self.__logger.info(bucket.predictions)
        self.__logger.info(bucket.label_ids)
        self.__logger.info(bucket.__doc__)
