import logging

import torch
import numpy as np
import src.structures.bert.parameters

import config


class Preview:

    def __init__(self):
        """

        """

        self.__tokenizer = src.structures.bert.parameters.Parameters().tokenizer

        # A random number generator instance
        self.__rng = np.random.default_rng(seed=config.Config().seed)

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)
        
    @staticmethod
    def __content(segment: dict):
        """

        :param segment:
        :return:
        """

        inputs_: torch.Tensor = segment['input_ids']
        labels_: torch.Tensor = segment['labels']
        token_type_identifiers_: torch.Tensor = segment['token_type_ids']
        attention_mask_: torch.Tensor = segment['attention_mask']
        offset_mapping_: torch.Tensor = segment['offset_mapping']
        
        return inputs_, labels_, token_type_identifiers_, attention_mask_, offset_mapping_

    def __details(self, name: str, item: torch.Tensor):
        """

        :param name:
        :param item:
        :return:
        """

        self.__logger.info('%s: %s', name, item.shape)
        self.__logger.info(item.data)

    def exc(self, dataset):
        """

        :param dataset:
        :return:
        """

        # A segment of dataset
        index = self.__rng.integers(low=0, high=(dataset.__len__() - 1))
        segment: dict = dataset.__getitem__(index)

        self.__logger.info('Previewing an instance of the data: ...')
        self.__logger.info(segment.keys())
        
        # The content of the segment
        inputs_, labels_, token_type_identifiers_, _, offset_mapping_ = self.__content(segment=segment)
        self.__details(name='inputs', item=inputs_)
        self.__details(name='labels', item=labels_)
        self.__details(name='token type identifiers', item=token_type_identifiers_)
        self.__details(name='offset mapping', item=offset_mapping_)
