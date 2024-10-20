"""Module structures.py"""
import pandas as pd
import torch.utils.data as tu
import transformers

import src.elements.arguments as ag
import src.elements.structures as sr
import src.elements.vault as vu
import src.models.distil.dataset
import src.models.loader


class Structures:
    """
    Class Structures<br>
    ----------------<br>

    Builds and delivers the data structures per modelling stage
    """

    def __init__(self, enumerator: dict, arguments: ag.Arguments, vault: vu.Vault,
                 tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase):
        """

        :param enumerator: Of tags; key &rarr; identifier, value &rarr; label<br>
        :param arguments: A suite of values/arguments for machine learning model development.<br>
        :param vault: An object of dataframes, consisting of the training, validating, and testing data sets.<br>
        :param tokenizer: The tokenizer of text.<br>
        """

        # A set of values, and data, for machine learning model development
        self.__enumerator = enumerator
        self.__arguments = arguments
        self.__vault = vault
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

        return self.__structure(frame=self.__vault.training, parameters=parameters)

    def validating(self) -> sr.Structures:
        """
        Delivers the validation data's Dataset & DataLoader

        :return:
        """

        # Modelling parameters
        parameters = {'batch_size': self.__arguments.VALID_BATCH_SIZE,
                      'shuffle': True, 'num_workers': 0}

        return self.__structure(frame=self.__vault.validating, parameters=parameters)

    def testing(self) -> sr.Structures:
        """
        Delivers the testing data's Dataset & DataLoader

        :return:
        """

        # Modelling parameters
        parameters = {'batch_size': self.__arguments.TEST_BATCH_SIZE,
                      'shuffle': True, 'num_workers': 0}

        return self.__structure(frame=self.__vault.testing, parameters=parameters)
