"""Module hyperspace.py"""
import json
import logging

import config
import src.elements.hyperspace as hp
import src.elements.s3_parameters as s3p
import src.elements.service as sr
import src.s3.unload


class Hyperspace:

    def __init__(self, service: sr.Service, s3_parameters: s3p.S3Parameters):
        """

        :param service: A suite of services for interacting with Amazon Web Services.
        :param s3_parameters: The overarching S3 (Simple Storage Service) parameters
                              settings of this project, e.g., region code name, buckets, etc.
        """

        self.__service: sr.Service = service
        self.__s3_parameters = s3_parameters

        # Configurations
        self.__configurations = config.Config()

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def __get_dictionary(self, node: str) -> dict:
        """
        s3:// {bucket.name} / {prefix.root} + {prefix.name} / {key.name}

        :param node: {prefix.name} / {key.name}
        :return:
        """

        key_name = self.__s3_parameters.path_internal_configurations + node

        buffer = src.s3.unload.Unload(service=self.__service).exc(
            bucket_name=self.__s3_parameters.internal, key_name=key_name)
        dictionary = json.loads(buffer)

        return dictionary

    def exc(self, node: str) -> hp.Hyperspace:
        """
        s3:// {bucket.name} / {prefix.root} + {prefix.name} / {key.name}

        :param node: {prefix.name} / {key.name}
        :return:
        """

        dictionary = self.__get_dictionary(node=node)

        items = {'learning_rate_distribution': dictionary['continuous']['learning_rate'],
                 'weight_decay_distribution': dictionary['continuous']['weight_decay'],
                 'weight_decay_choice': dictionary['choice']['weight_decay'],
                 'per_device_train_batch_size': dictionary['choice']['per_device_train_batch_size']}

        hyperspace = hp.Hyperspace(**items)

        return hyperspace
