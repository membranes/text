import pandas as pd

import src.elements.variable

class Steps:

    def __init__(self, enumerator: dict, archetype: dict,
                 training: pd.DataFrame, validating: pd.DataFrame):
        """

        :param enumerator:
        :param archetype:
        :param training:
        :param validating:
        """

        # Inputs
        self.__enumerator = enumerator
        self.__archetype = archetype
        self.__training = training
        self.__validating = validating

        # A set of values for machine learning model development
        self.__variable = src.elements.variable.Variable()
        self.__variable = self.__variable._replace(EPOCHS=2, TRAIN_BATCH_SIZE=16, VALID_BATCH_SIZE=16)

