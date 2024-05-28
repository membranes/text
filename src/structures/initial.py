import pandas as pd

import src.structures.bert.data
import src.types.variable

class Initial:

    def __init__(self, frame: pd.DataFrame, enumerator: dict) -> None:

        self.__frame = frame
        self.__enumerator = enumerator
        self.__fraction = 0.8
        self.__seed = 5

        self.__variable = src.types.variable.Variable()

    def __split(self):

        blob = self.__frame.copy()

        training = blob.sample(frac=self.__fraction, random_state=self.__seed)
        validating = blob.drop(training.index)

        training.reset_index(drop=True, inplace=True)
        validating.reset_index(drop=True, inplace=True)

        return training, validating
    
    def __bert(self, blob: pd.DataFrame):

        return src.structures.bert.data.Data(
            blob, self.__variable, self.__enumerator)


    def exc(self):

        training, validating = self.__split()

        btr = self.__bert(blob=training)
        bva = self.__bert(blob=validating)

        print(btr[0])
        print(bva[0])
