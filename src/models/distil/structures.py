
import transformers

import src.elements.frames as fr
import src.elements.variable as vr
import src.models.loadings


class Structures:

    def __init__(self, enumerator: dict, variable: vr.Variable, frames: fr.Frames,
                 tokenizer: transformers.tokenization_utils_base):
        """

        :param enumerator:
        :param variable:
        :param frames:
        """

        # A set of values, and data, for machine learning model development
        self.__enumerator = enumerator
        self.__variable = variable
        self.__frames = frames

        self.__tokenizer = tokenizer

        # For DataLoader creation
        self.__loadings = src.models.loadings.Loadings()
