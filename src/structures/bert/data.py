import numpy as np
import pandas as pd
import torch
import torch.utils.data

import src.elements.variable
import src.structures.bert.parameters


class Data(torch.utils.data.Dataset):

    def __init__(self, frame: pd.DataFrame, variable: src.elements.variable.Variable,
                 enumerator: dict) -> None:
        """

        :param frame:
        :param variable:
        :param enumerator:
        """

        super().__init__()

        self.__frame = frame
        self.__length = len(self.__frame)

        self.__variable = variable
        self.__enumerator = enumerator
        self.__tokenizer = src.structures.bert.parameters.Parameters().tokenizer

    @staticmethod
    def __temporary(encoding: dict, classes: np.ndarray, codes: list) -> np.ndarray:
        """

        :param encoding:
        :param classes:
        :param codes:
        :return:
        """

        temporary = classes.copy()

        i = 0
        for index, mapping in enumerate(encoding['offset_mapping']):

            if mapping[0] == 0 and mapping[1] != 0:
                temporary[index] = codes[i]
                i += 1

        return temporary

    @staticmethod
    def __structure(encoding: dict, temporary: np.ndarray) -> dict:
        """

        :param encoding:
        :param temporary:
        :return: A dictionary of tensors
        """

        item = {key: torch.as_tensor(value) for key, value in encoding.items()} 
        item['labels'] = torch.as_tensor(temporary)

        return item 

    def __getitem__(self, index) -> dict:
        """

        :param index: A row index
        :return:
        """

        # A sentence's words, and the tokenization of words
        words: list[str] = self.__frame['sentence'][index].strip().split()
        encoding: dict = self.__tokenizer(words, padding='max_length', truncation=True,
                                          max_length=self.__variable.MAX_LENGTH, return_offsets_mapping=True)
        classes: np.ndarray = np.ones(shape=len(encoding['offset_mapping']), dtype=int) * -100

        # The corresponding tags of a sentence's words, and the code of each tag
        tags: list[str] = self.__frame['tagstr'][index].split(',')
        codes = [self.__enumerator[tag] for tag in tags]

        # Re-setting
        temporary: np.ndarray = self.__temporary(encoding=encoding, classes=classes, codes=codes)

        return self.__structure(encoding=encoding, temporary=temporary)
    
    def __len__(self):
        """

        :return:
        """

        return self.__length
