import numpy as np
import pandas as pd
import torch
import torch.utils.data
import transformers

import src.elements.variable


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

        self.__tokenizer = transformers.BertTokenizerFast.from_pretrained(
            pretrained_model_name_or_path='google-bert/bert-base-uncased')

    @staticmethod
    def __temporary(encoding: dict, classes: np.ndarray, codes: list) -> np.ndarray:
        """

        :param encoding:
        :param classes:
        :param codes:
        :return:
        """

        i = 0
        for index, mapping in enumerate(encoding['offset_mapping']):

            if mapping[0] == 0 and mapping[1] != 0:
                classes[index] = codes[i]
                i += 1

        return classes

    @staticmethod
    def __space(encoding: dict, classes: np.ndarray) -> dict:
        """

        :param encoding:
        :param classes:
        :return:
        """

        item = {key: torch.as_tensor(value) for key, value in encoding.items()} 
        item['labels'] = torch.as_tensor(classes)     

        return item 

    def __getitem__(self, index):
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
        classes: np.ndarray = self.__temporary(encoding=encoding, classes=classes, codes=codes)

        return self.__space(encoding=encoding, classes=classes)
    
    def __len__(self):
        """

        :return:
        """

        return self.__length
