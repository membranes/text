"""Module interface.py"""
import logging

import pandas as pd

import src.elements.frames
import src.models.bert.steps
import src.models.distil.steps
import src.models.splittings
import src.elements.arguments as ag


class Interface:
    """
    An interface to each model.
    """

    def __init__(self, data: pd.DataFrame, enumerator: dict, archetype: dict):
        """

        :param data: The data set for the training, validating, etc., stages
        :param enumerator: The tags and their identification codes.
        :param archetype: The inverse dict of enumerator.
        """

        self.__training, self.__validating, self.__testing = src.models.splittings.Splittings().exc(data=data)
        self.__frames = src.elements.frames.Frames(
            training=self.__training, validating=self.__validating, testing=self.__testing)
        self.__enumerator = enumerator
        self.__archetype = archetype

    def exc(self, architecture: str, arguments: ag.Arguments) -> None:
        """

        :param architecture:
        :param arguments:
        :return:
        """

        match architecture:
            case 'bert':
                src.models.bert.steps.Steps(
                    enumerator=self.__enumerator, archetype=self.__archetype, frames=self.__frames).exc()
            case 'electra':
                logging.info('ELECTRA: Future')
            case 'roberta':
                logging.info(self.__frames.training)
                logging.info('ROBERTA: Future')
            case 'distil':
                src.models.distil.steps.Steps(
                    enumerator=self.__enumerator, archetype=self.__archetype, arguments=arguments, frames=self.__frames).exc()
            case 'ensemble':
                logging.info('BiLSTM, BERT, & CRF: Future\nhttps://link.springer.com/article/10.1007/s42979-024-02835-z')
            case _:
                logging.info('Unknown architecture')
