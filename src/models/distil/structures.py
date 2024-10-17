"""Module structures.py"""
import pandas as pd
import torch.utils.data as tu
import transformers

import src.elements.arguments as ag
import src.elements.frames as fr
import src.elements.structures as sr
import src.models.distil.dataset
import src.models.loader


class Structures:

    def __init__(self, enumerator: dict, arguments: ag.Arguments, frames: fr.Frames,
                 tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase):
        """

        :param enumerator:
        :param arguments:
        :param frames:
        :param tokenizer:
        """

        # A set of values, and data, for machine learning model development
        self.__enumerator = enumerator
        self.__arguments = arguments
        self.__frames = frames

        self.__tokenizer = tokenizer

        # For DataLoader creation
        self.__loader = src.models.loader.Loader()

    def __structure(self, frame: pd.DataFrame, parameters: dict) -> sr.Structures:
        """

        :param frame:
        :param parameters:
        :return:
        """

        dataset = src.models.distil.dataset.Dataset(
            frame=frame, enumerator=self.__enumerator, tokenizer=self.__tokenizer)
        dataloader: tu.DataLoader = self.__loader.exc(dataset=dataset, parameters=parameters)

        return sr.Structures(dataset=dataset, dataloader=dataloader)

    def training(self) -> sr.Structures:
        """
        Delivers the training data's Dataset & DataLoader

        :return:
        """

        # Modelling parameters
        parameters = {'batch_size': self.__arguments.TRAIN_BATCH_SIZE,
                      'shuffle': True, 'num_workers': 0}

        return self.__structure(frame=self.__frames.training, parameters=parameters)

    def validating(self) -> sr.Structures:
        """
        Delivers the validation data's Dataset & DataLoader

        :return:
        """

        # Modelling parameters
        parameters = {'batch_size': self.__arguments.VALID_BATCH_SIZE,
                      'shuffle': True, 'num_workers': 0}

        return self.__structure(frame=self.__frames.validating, parameters=parameters)

    def testing(self) -> sr.Structures:
        """
        Delivers the testing data's Dataset & DataLoader

        :return:
        """

        # Modelling parameters
        parameters = {'batch_size': self.__arguments.TEST_BATCH_SIZE,
                      'shuffle': True, 'num_workers': 0}

        return self.__structure(frame=self.__frames.testing, parameters=parameters)
