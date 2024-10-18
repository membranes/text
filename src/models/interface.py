"""Module interface.py"""
import logging

import pandas as pd

import src.elements.vault
import src.models.bert.steps
import src.models.distil.steps
import src.models.splittings
import src.elements.arguments as ag
import src.elements.hyperspace as hp


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
        self.__vault = src.elements.vault.Vault(
            training=self.__training, validating=self.__validating, testing=self.__testing)
        self.__enumerator = enumerator
        self.__archetype = archetype

    def exc(self, architecture: str, arguments: ag.Arguments, hyperspace: hp.Hyperspace) -> None:
        """

        :param architecture:
        :param arguments:
        :param hyperspace:
        :return:
        """

        match architecture:
            case 'bert':
                src.models.bert.steps.Steps(
                    enumerator=self.__enumerator, archetype=self.__archetype,
                    arguments=arguments, hyperspace=hyperspace, vault=self.__vault).exc()
            case 'electra':
                logging.info('ELECTRA: Future')
            case 'roberta':
                logging.info(self.__vault.training)
                logging.info('ROBERTA: Future')
            case 'distil':
                src.models.distil.steps.Steps(
                    enumerator=self.__enumerator, archetype=self.__archetype,
                    arguments=arguments, hyperspace=hyperspace, vault=self.__vault).exc()
            case 'ensemble':
                logging.info('BiLSTM, BERT, & CRF: Future\nhttps://link.springer.com/article/10.1007/s42979-024-02835-z')
            case _:
                logging.info('Unknown architecture')
