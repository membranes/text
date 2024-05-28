import logging

import src.structures.bert.parameters


class Preview:

    def __init__(self):

        self.__tokenizer = src.structures.bert.parameters.Parameters().tokenizer

        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def exc(self, dataset):

        index = dataset.__len__() - 1
        dictionary = dataset.__getitem__(index)

        self.__logger.info(dictionary)

        for input in dictionary['input_ids']:
            self.__logger.info(input)

        '''
        tokens = self.__tokenizer.convert_ids_to_tokens(dataset[0]['input_ids'])
        labels = dataset[0]['labels']
        for token, label in zip(tokens, labels):

            self.__logger.info('%s: %s', token, label)
        '''

