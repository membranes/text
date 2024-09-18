"""Module metrics.py"""
import logging
import numpy as np
import evaluate
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

    def exc(self, bucket: transformers.trainer_utils.PredictionOutput):
        """
        logging.info('Determining active labels & predictions')
        active = np.not_equal(labels, -100)

        true_labels = labels[active]
        true_predictions = predictions[active]

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
        logging.info(true_predictions)
        logging.info(true_labels)
        results = self.__seqeval.compute(predictions=true_predictions, references=true_labels, zero_division=0.0)

        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
        }
