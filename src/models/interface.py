
import pandas as pd

import src.models.splittings
import src.models.bert.steps


class Interface:

    def __init__(self, frame: pd.DataFrame, enumerator: dict, archetype: dict):
        """

        :param frame: The data set for the training, validating, etc., stages
        :param enumerator:
        :param archetype:
        """

        self.__training, self.__validating, _ = src.models.splittings.Splittings(frame=frame).exc()
        self.__enumerator = enumerator
        self.__archetype = archetype

    def exc(self) -> None:
        """

        :return:
        """

        # bert
        src.models.bert.steps.Steps(enumerator=self.__enumerator, archetype=self.__archetype,
                                    training=self.__training, validating=self.__validating).exc()

        # electra
        # src.structures.electra.steps

        # distil
        # src.structure.distil.steps

        # Transfer Learning with BiLSTM, BERT and CRF
        # https://link.springer.com/article/10.1007/s42979-024-02835-z
