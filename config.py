"""config.py"""
import os


class Config:
    """
    Config
    """

    def __init__(self) -> None:
        """
        Constructor

        Variables denoting a path - including or excluding a filename - have an underscore
        """

        self.warehouse = os.path.join(os.getcwd(), 'warehouse')
        self.artefacts_ = os.path.join(self.warehouse, 'artefacts')

        # The prepared data s3:// {bucket.name} / {prefix.root} + {prefix.name} / {key.name}
        # Herein
        #   * node ≡ {prefix.name} / {key.name}
        self.data_ = 'prepared/data.csv'
        self.enumerator_ = 'prepared/enumerator.json'
        self.archetype_ = 'prepared/archetype.json'

        # A template of Amazon S3 (Simple Storage Service) parameters
        self.s3_parameters_template = 'https://raw.githubusercontent.com/membranes/.github/refs/heads/master/profile/s3_parameters.yaml'


        '''
        Delete after re-designing.
        '''

        self.datapath = os.path.join(os.getcwd(), 'data')

        # Addressing category imbalance ...
        self.minimum_category_frequency = 1000

        # Splitting fractions:
        # * 80% of the data for training
        # * A tenth of the remaining 20% for testing, i.e., 18%
        #   for validating, 2% for testing
        self.fraction = 0.8
        self.aside = 0.1

        # Seed: All cases
        self.seed = 5
