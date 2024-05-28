import logging

import torch

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
        dictionary: dict = dataset.__getitem__(index)
        elements: torch.Tensor = dictionary['input_ids']
        labels: torch.Tensor = dictionary['labels']

        self.__logger.info(dictionary.keys())
        self.__logger.info(dictionary)

        self.__logger.info(elements.shape)
        self.__logger.info(elements)

        self.__logger.info(labels.shape)
        self.__logger.info(labels)


        for element, label in zip(elements[:5], labels[:5]):
            self.__logger.info(element)
            self.__logger.info(label)
            self.__logger.info(self.__tokenizer.convert_ids_to_tokens(element))
