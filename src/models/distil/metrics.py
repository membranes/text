"""Mudule metrics.py"""
import logging
import numpy as np
import evaluate
import transformers.trainer_utils


class Metrics:
    """

    https://huggingface.co/spaces/evaluate-metric/seqeval/blob/main/seqeval.py
    """

    def __init__(self):
        """

        """

        self.__seqeval = evaluate.load('seqeval')

    def exc(self, bucket: transformers.trainer_utils.PredictionOutput):
        """

        :param bucket:
        :return:
        """

        predictions = bucket.predictions
        predictions = np.argmax(predictions, axis=2)
        labels = bucket.label_ids

        # Active
        logging.info('Determining active labels & predictions')
        active = np.not_equal(labels, -100)

        true_labels = labels[active]
        true_predictions = predictions[active]

        results = self.__seqeval.compute(predictions=true_predictions, references=true_labels)

        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
        }
