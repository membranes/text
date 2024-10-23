"""Module transfer.py"""
import glob
import os

import config
import src.data.dictionary
import src.elements.s3_parameters as s3p
import src.elements.service as sr
import src.s3.ingress
import src.functions.directories


class Transfer:
    """
    Transfers data files to an Amazon S3 (Simple Storage Service) prefix.

    """

    def __init__(self, service: sr.Service,  s3_parameters: s3p, architecture: str):
        """

        :param service: A suite of services for interacting with Amazon Web Services.
        :param s3_parameters: The overarching S3 parameters settings of this
                              project, e.g., region code name, buckets, etc.
        :param architecture: The pre-trained model architecture in focus.
        """

        self.__service: sr.Service = service
        self.__s3_parameters: s3p.S3Parameters = s3_parameters
        self.__architecture: str = architecture
        self.__configurations = config.Config()

        self.__dictionary = src.data.dictionary.Dictionary(architecture=architecture)
        self.__directories = src.functions.directories.Directories()

    @staticmethod
    def __name(pathstr: str):
        """

        :param pathstr:
        :return:
        """

        left = pathstr.split('_', maxsplit=4)
        right = pathstr.rsplit('_', maxsplit=2)
        strings = left[1:3] + right[-2:]

        name = '_'.join(strings)

        return name

    def __runs(self):
        """
        Deletes the runs directories of the hyperparameter search stage.

        :return:
        """

        path: str = os.path.join(self.__configurations.artefacts_, self.__architecture, 'hyperparameters', 'run*')
        directories = glob.glob(pathname=path, recursive=True)

        for directory in directories:
            self.__directories.cleanup(directory)

    def __renaming(self):
        """
        Renames the objective directories because their default names are too long.

        :return:
        """

        # The directories that start with _objective; add a directory check
        elements = glob.glob(pathname=os.path.join(self.__configurations.artefacts_, '**', '_objective*'), recursive=True)
        directories = [element for element in elements if os.path.isdir(element)]

        # Bases
        bases = [os.path.basename(directory) for directory in directories]
        bases = [self.__name(base) for base in bases]

        # Endpoints
        endpoints = [os.path.dirname(directory) for directory in directories]

        for directory, base, endpoint in zip(directories, bases, endpoints):
            os.rename(src=directory, dst=os.path.join(endpoint, base))

    def exc(self):
        """

        :return:
        """

        self.__runs()

        self.__renaming()

        strings = self.__dictionary.exc(
            path=self.__configurations.artefacts_, extension='*', prefix=self.__s3_parameters.path_internal_artefacts)

        messages = src.s3.ingress.Ingress(
            service=self.__service, bucket_name=self.__s3_parameters.internal).exc(strings=strings)

        return messages
