"""Module metrics.py"""
import collections
import logging

import evaluate
import numpy as np
import transformers.trainer_utils


class Metrics:
    """

    https://huggingface.co/spaces/evaluate-metric/seqeval/blob/main/seqeval.py
    """

    def __init__(self, archetype: dict):
        """

        :param archetype:
        """

        self.__archetype = archetype
        self.__seqeval = evaluate.load('seqeval')

        # Logging
        logging.basicConfig(level=logging.INFO,
                        format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                        datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def __restructure(self, key: str, dictionary: dict):
        """

        :param key:
        :param dictionary:
        :return:
        """

        return {f'{key}_{k}': v for k, v in dictionary.items()}

    def __decompose(self, metrics: dict) -> dict:
        """

        :param metrics:{<br>
                    &nbsp; 'class<sub>1</sub>': {'metric<sub>1</sub>': value, 'metric<sub>2</sub>': value, ...},<br>
                    &nbsp; 'class<sub>2</sub>': {'metric<sub>1</sub>': value, 'metric<sub>2</sub>': value, ...}, ...}
        :return:
        """

        # Class level metrics
        disaggregates = {k: v for k, v in metrics.items() if not k.startswith('overall')}

        # Re-structuring the dictionary of class level metrics
        metrics_per_class = list(map(lambda x: self.__restructure(x[0], x[1]), disaggregates.items()))

        # Overarching metrics
        aggregates = {k: v for k, v in metrics.items() if k.startswith('overall')}

        return dict(collections.ChainMap(*metrics_per_class, aggregates))

    def exc(self, bucket: transformers.trainer_utils.PredictionOutput):
        """
        logging.info('Determining active labels & predictions')
        active = np.not_equal(labels, -100)

        :param bucket:
        :return:
        """

        predictions = bucket.predictions
        predictions = np.argmax(predictions, axis=2)
        labels = bucket.label_ids

        # Or
        true_predictions = [
            [self.__archetype[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.__archetype[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        # Hence
        metrics = self.__seqeval.compute(predictions=true_predictions, references=true_labels, zero_division=0.0)
        self.__logger.info('The original metrics structure:\n%s', metrics)

        decomposition = self.__decompose(metrics=metrics)
        self.__logger.info('The restructured dictionary of metrics:\n%s', metrics)

        return decomposition
