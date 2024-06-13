
import pandas as pd

import src.structures.initial
import src.structures.bert.steps


class Interface:

    def __init__(self, frame: pd.DataFrame, enumerator: dict):

        self.__training, self.__validating = src.structures.initial.Initial(frame=frame).exc()
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
