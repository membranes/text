import numpy as np


class Numerics:

    def __init__(self, originals: list, predictions: list):

        self.__originals = np.array(originals)
        self.__predictions = np.array(predictions)
