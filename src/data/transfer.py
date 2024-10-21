
import src.data.dictionary
import src.s3.ingress

import src.elements.s3_parameters as s3p
import src.elements.service as sr

import config


class Transfer:

    def __init__(self, service: sr.Service,  s3_parameters: s3p, architecture: str):
        """

        :param service: A suite of services for interacting with Amazon Web Services.
        :param s3_parameters: The overarching S3 parameters settings of this
                              project, e.g., region code name, buckets, etc.
        """

        self.__service: sr.Service = service
        self.__s3_parameters: s3p.S3Parameters = s3_parameters
        self.__configurations = config.Config()

        self.__dictionary = src.data.dictionary.Dictionary(architecture=architecture)

    def exc(self):
        """

        :return:
        """

        strings = self.__dictionary.exc(
            path=self.__configurations.warehouse, extension='*', prefix=self.__s3_parameters.path_internal_artefacts)

        messages = src.s3.ingress.Ingress(
            service=self.__service, bucket_name=self.__s3_parameters.internal).exc(strings=strings)

        return messages
