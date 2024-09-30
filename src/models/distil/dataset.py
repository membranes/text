import logging

import pandas as pd
import torch.utils.data
import transformers

import src.elements.variable as vr


class Dataset(torch.utils.data.Dataset):

    def __init__(self, frame: pd.DataFrame, variable: vr.Variable, enumerator: dict,
                 tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase):
        """

        :param frame:
        :param tokenizer:
        """

        super().__init__()

        self.__frame = frame
        self.__length = self.__frame.shape[0]

        self.__variable = variable
        self.__enumerator = enumerator
        self.__tokenizer = tokenizer

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def __getitem__(self, index):

        words: list[str] = self.__frame['sentence'][index].strip().split()
        encodings = self.__tokenizer(words, truncation=True, is_split_into_words=True, padding='max_length', max_length=self.__length)

        tags: list[str] = self.__frame['tagstr'][index].split(',')
        numerals: list[int] = [self.__enumerator[tag] for tag in tags]

        identifiers = encodings.word_ids()

        previous = None
        register = []
        for identifier in identifiers:
            if identifier is None:
                register.append(-100)
            elif identifier != previous:
                register.append(numerals[identifier])
            else:
                register.append(-100)
            previous = identifier

        encodings['labels'] = register

        item = {key: torch.as_tensor(value) for key, value in encodings.items()}

        return item

    def __len__(self):
        """

        :return:
        """

        return self.__length
