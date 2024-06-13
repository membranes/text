"""config.py"""
import os


class Config:
    """
    Config
    """

    def __init__(self) -> None:
        """
        Constructor
        """

        self.datapath = os.path.join(os.getcwd(), 'data')
        self.warehouse = os.path.join(os.getcwd(), 'warehouse')

        # Addressing category imbalance ...
        self.minimum_category_frequency = 1000

        '''Modelling'''

        # Splitting fractions
        self.fraction = 0.8
        self.aside = 0.05

        # Seed: All cases
        self.seed = 5
