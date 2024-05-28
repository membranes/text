import logging

import pandas as pd
import torch.utils.data

import src.elements.variable
import src.structures.bert.data
import src.structures.bert.preview


class Initial:

    def __init__(self, frame: pd.DataFrame, enumerator: dict) -> None:

        self.__frame = frame
        self.__enumerator = enumerator
        self.__fraction = 0.8
        self.__seed = 5

        self.__variable = src.elements.variable.Variable()
        self.__preview = src.structures.bert.preview.Preview()

        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def __split(self):

        blob = self.__frame.copy()

        training = blob.sample(frac=self.__fraction, random_state=self.__seed)
        validating = blob.drop(training.index)

        training.reset_index(drop=True, inplace=True)
        validating.reset_index(drop=True, inplace=True)

        return training, validating
    
    def __bert(self, blob: pd.DataFrame) -> dict:

        dataset = src.structures.bert.data.Data(blob, self.__variable, self.__enumerator)
        self.__logger.info(type(dataset))
        self.__preview.exc(dataset=dataset)

        return dataset

    def exc(self):

        training, validating = self.__split()

        btr = self.__bert(blob=training)
        bva = self.__bert(blob=validating)
