import numpy as np

import sklearn.metrics as sm

import dask


class Numerics:

    def __init__(self, originals: list[str], predictions: list[str]):
        """

        :param originals: The list of original labels
        :param predictions: The list of predicted labels
        """

        self.__originals: np.ndarray = np.array(originals)
        self.__predictions: np.ndarray = np.array(predictions)

    def __measures(self, name: str):
        """

        :param name: The name of one of the labels
        :return:
        """

        _true: np.ndarray = (self.__originals == name).astype(int)
        _prediction: np.ndarray =  (self.__predictions == name).astype(int)

        tn, fp, fn, tp = sm.confusion_matrix(
            y_true=_true, y_pred=_prediction).ravel()

        return {name: {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}}

    def exc(self):

        names = np.unique(self.__originals)
        objects = [dask.delayed(self.__measures)(name) for name in names]
        calculations = dask.compute(objects)[0]

        structure = {k: v for c in calculations for k, v in c.items()}

        return structure
