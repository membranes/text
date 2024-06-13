
import pandas as pd

import src.structures.splittings
import src.structures.bert.steps


class Interface:

    def __init__(self, frame: pd.DataFrame, enumerator: dict):
        """

        :param frame: The data set for the training, validating, etc., stages
        :param enumerator:
        """

        self.__training, self.__validating, _ = src.structures.splittings.Splittings(frame=frame).exc()
        self.__enumerator = enumerator

    def exc(self) -> None:
        """

        :return:
        """

        # bert
        src.structures.bert.steps.Steps(
            enumerator=self.__enumerator, training=self.__training, validating=self.__validating).exc()

        # electra
        # src.structures.electra.steps

        # distil
        # src.structure.distil.steps

        # Transfer Learning with BiLSTM, BERT and CRF
        # https://link.springer.com/article/10.1007/s42979-024-02835-z
