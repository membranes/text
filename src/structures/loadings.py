

import typing

import torch.utils.data as tu

import src.elements.variable as va


class Loadings:
    """
    Description
    ===========

    For creating DataLoaders.  The Steps class of a model calls this
    class for creating DataLoaders.
    """

    def __init__(self, variable: va.Variable,  training: tu.Dataset, validating: tu.Dataset = None,
                 testing: tu.Dataset = None):
        """

        :param variable:
        :param training:
        :param validating:
        :param testing:
        """

        self.__variable = variable

        self.__training = training
        self.__validating = validating
        self.__testing = testing

    def __training_loader(self) -> tu.DataLoader:
        """

        :return:
        """

        parameters = {'batch_size': self.__variable.TRAIN_BATCH_SIZE,
                      'shuffle': True, 'num_workers': 0}

        return tu.DataLoader(self.__training, **parameters)

    def __validating_loader(self) -> tu.DataLoader:
        """

        :return:
        """

        parameters = {'batch_size': self.__variable.VALID_BATCH_SIZE,
                      'shuffle': True, 'num_workers': 0}

        return tu.DataLoader(self.__validating, **parameters)

    def exc(self) -> typing.Tuple[tu.DataLoader, tu.DataLoader]:
        """

        :return:
        """

        return self.__training_loader(), self.__validating_loader()
