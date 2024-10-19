"""Module arguments.py"""
import os
import logging
import pandas as pd

import config
import src.elements.s3_parameters as s3p
import src.elements.arguments



class Arguments:
    """
    Arguments<br>
    ---------<br>

    Reads-in a JSON (JavaScript Object Notation) file of arguments
    """

    def __init__(self, s3_parameters: s3p.S3Parameters):
        """

        :param s3_parameters: s3_parameters: The overarching S3 (Simple Storage Service) parameters
                              settings of this project, e.g., region code name, buckets, etc.
        """

        self.__s3_parameters = s3_parameters

        # Configurations
        self.__configurations = config.Config()

        # Logging
        logging.basicConfig(level=logging.INFO,
                        format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                        datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def __get_dictionary(self, node: str):
        """
        s3:// {bucket.name} / {prefix.root} + {prefix.name} / {key.name}

        :param node: {prefix.name} / {key.name}
        :return:
        """

        path = 's3://' + self.__s3_parameters.internal + '/' + self.__s3_parameters.path_internal_configurations + node
        self.__logger.info(path)

        try:
            values = pd.read_json(path_or_buf=path, orient='index')
        except ImportError as err:
            raise err from err

        return values.to_dict()[0]

    def exc(self, node: str) -> src.elements.arguments.Arguments:
        """
        s3:// {bucket.name} / {prefix.root} + {prefix.name} / {key.name}

        :param node: {prefix.name} / {key.name}
        :return:
        """

        # Get the dictionary of arguments
        dictionary = self.__get_dictionary(node=node)

        # Set up the model output directory parameter
        model_output_directory = os.path.join(self.__configurations.warehouse, dictionary['name'])
        dictionary['model_output_directory'] = model_output_directory

        return src.elements.arguments.Arguments(**dictionary)
