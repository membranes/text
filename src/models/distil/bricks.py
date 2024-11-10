import datasets
import transformers

import src.elements.vault as vu

class Bricks:

    def __init__(self, vault: vu.Vault, tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase):

        self.__vault = vault
        self.__tokenizer = tokenizer

    def __splittings(self):

        splittings = datasets.DatasetDict({
            'training': datasets.Dataset.from_pandas(self.__vault.training),
            'validating': datasets.Dataset.from_pandas(self.__vault.validating),
            'testing': datasets.Dataset.from_pandas(self.__vault.testing)
        })

        return splittings

    def __tokenize(self, feeds):
        """

        :param feeds: sentence_identifier, sentence, tagstr
        :return:
        """

        # tokenization of words
        inputs = self.__tokenizer(feeds['sentence'], truncation=True, is_split_into_words=True, padding='max_length')

        # A placeholder for labels, i.e., ...
        labels = []

        # iteration code, [list of tags]
        for iteration, tags in enumerate(feeds['tagstr']):

            # word identifiers
            word_identifiers = inputs.word_ids(batch_index=iteration)

            previous = None
            label_identifiers = []
            for word_identifier in word_identifiers:
                if word_identifier is None:
                    label_identifiers.append(-100)




    def exc(self):

        splittings = self.__splittings()
