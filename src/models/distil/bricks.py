import datasets
import transformers
import pandas as pd

import src.elements.vault as vu

class Bricks:

    def __init__(self, vault: vu.Vault, tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase):
        """

        :param vault:
        :param tokenizer:
        """

        self.__vault = vault
        self.__tokenizer = tokenizer

        # The critical fields
        self.__fields = ['sentence_identifier', 'words', 'codes']

    def __features(self, blob: pd.DataFrame) -> pd.DataFrame:
        """

        :param blob:
        :return:
        """

        frame = blob.copy()
        frame['words'] = frame['sentence'].str.split()
        frame['codes'] = frame['code_per_tag'].str.split(',').apply(
            lambda x: list(map(int, x))
        )

        return frame[self.__fields]

    def __splittings(self):

        splittings = datasets.DatasetDict({
            'training': datasets.Dataset.from_pandas(
                self.__features(self.__vault.training)),
            'validating': datasets.Dataset.from_pandas(
                self.__features(self.__vault.validating)),
            'testing': datasets.Dataset.from_pandas(
                self.__features(self.__vault.testing))
        })

        return splittings

    def __tokenize(self, feeds):
        """

        :param feeds: sentence_identifier, sentence, tagstr
        :return:
        """

        # tokenization of words
        inputs = self.__tokenizer(feeds['words'], truncation=True, is_split_into_words=True, padding='max_length')

        # A placeholder for labels, i.e., codes
        labels = []

        # iteration number, a codes list
        for iteration, codes in enumerate(feeds['codes']):

            # word identifiers
            word_identifiers = inputs.word_ids(batch_index=iteration)

            # previous word identifier
            previous = None

            # A placeholder for label identifiers
            label_identifiers = []
            for word_identifier in word_identifiers:
                if word_identifier is None:
                    label_identifiers.append(-100)
                elif word_identifier != previous:
                    label_identifiers.append(codes[word_identifier])
                else:
                    label_identifiers.append(-100)
                previous = word_identifier

            # Hence
            labels.append(label_identifiers)

            # Therefore
            inputs['labels'] = labels

            return  inputs




    def exc(self):

        splittings = self.__splittings()
        yields = splittings.map(self.__tokenize, batched=True)
