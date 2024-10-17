"""Module algorithm.py"""
import sys

import src.models.bert.intelligence
import src.models.distil.intelligence

import src.elements.arguments as ag


class Algorithm:

    def __init__(self, architecture: str):
        """

        :param architecture: The name of the architecture in focus
        """

        self.__architecture = architecture

    def exc(self, arguments: ag.Arguments, enumerator: dict, archetype: dict):
        """

        :param arguments:
        :param enumerator:
        :param archetype:
        :return:
        """

        match self.__architecture:
            case 'bert':
                return src.models.bert.intelligence.Intelligence(
                    enumerator=enumerator, archetype=archetype, arguments=arguments)
            case 'electra':
                sys.exit('ELECTRA: Future')
            case 'roberta':
                sys.exit('ROBERTA: Future')
            case 'distil':
                return src.models.distil.intelligence.Intelligence(
                    enumerator=enumerator, archetype=archetype, arguments=arguments)
            case 'ensemble':
                sys.exit('BiLSTM, BERT, & CRF: Future\nhttps://link.springer.com/article/10.1007/s42979-024-02835-z')
            case _:
                sys.exit('Unknown architecture')
