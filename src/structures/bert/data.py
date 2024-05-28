import torch.utils.data
import torch
import transformers

import numpy as np
import pandas as pd

import src.elements.variable

class Data(torch.utils.data.Dataset):

    def __init__(self, frame: pd.DataFrame, variable: src.elements.variable.Variable,
                 enumerator: dict) -> None:
        super().__init__()

        self.__frame = frame
        self.__length = len(self.__frame)

        self.__variable = variable
        self.__enumerator = enumerator

        self.__tokenizer = transformers.BertTokenizerFast.from_pretrained(
            pretrained_model_name_or_path='google-bert/bert-base-uncased')

    def __temporary(self, encoding: dict, classes: np.ndarray, codes: list) -> np.ndarray:

        i = 0
        for index, mapping in enumerate(encoding['offset_mapping']):

            if mapping[0] == 0 and mapping[1] != 0:
                classes[index] = codes[i]
                i += 1

        return classes

    def __space(self, encoding: dict, classes: np.ndarray) -> dict:

        item = {key: torch.as_tensor(value) for key, value in encoding.items()} 
        item['labels'] = torch.as_tensor(classes)     

        return item 

    def __getitem__(self, index):
        """
        sentence: str, tagstr: str
        sentence.strip().split()
        tagstr.split(',')
        
        """

        # A sentence's words, and the tokenization of words
        words: list[str] = self.__frame['sentence'][index].strip().split()
        encoding: dict = self.__tokenizer(words, padding='max_length', 
                                    truncation=True, max_length=self.__variable.MAX_LENGTH, 
                                    return_offsets_mapping=True)        
        classes: np.ndarray = np.ones(shape=len(encoding['offset_mapping']), dtype=int) * -100

        # The corresponding taglets of a sentence's words, and the code of each taglet
        taglets: list[str] = self.__frame['tagstr'][index].split(',')
        codes = [self.__enumerator[label] for label in taglets]

        # Re-setting
        classes: np.ndarray = self.__temporary(encoding=encoding, classes=classes, codes=codes)

        return self.__space(encoding=encoding, classes=classes)
    
    def __len__(self):

        return self.__length
