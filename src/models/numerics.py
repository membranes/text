import numpy as np

import sklearn.metrics as sm


class Numerics:

    def __init__(self, originals: list[str], predictions: list[str]):

        self.__originals: np.ndarray = np.array(originals)
        self.__predictions: np.ndarray = np.array(predictions)

    def __measures(self, name: str):

        _true: np.ndarray = (self.__originals == name).astype(int)
        _prediction: np.ndarray =  (self.__predictions == name).astype(int)

        tn, fp, fn, tp = sm.confusion_matrix(
            y_true=_true, y_pred=_prediction).ravel()

        return {'name': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}}
