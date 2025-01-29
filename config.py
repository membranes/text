"""config.py"""
import os


class Config:
    """
    Config
    """

    def __init__(self) -> None:
        """
        Constructor<br>
        -----------<br>

        Variables denoting a path - including or excluding a filename - have an underscore suffix; this suffix is
        excluded for names such as warehouse, storage, depository, etc.<br><br>
        """

        self.warehouse = os.path.join(os.getcwd(), 'warehouse')
        self.artefacts_ = os.path.join(self.warehouse, 'artefacts')
        self.s3_parameters_key = 's3_parameters.yaml'
        self.architectures = ['bert', 'distil', 'roberta', 'electra']


        '''
        The Prepared Data
        
        s3:// {bucket.name} / {prefix.root} + {prefix.name} / {key.name}
        
        :param data_: {prefix.name} / {key.name}
        :param enumerator_: {prefix.name} / {key.name}
        :param archetype_: {prefix.name} / {key.name}
        '''
        self.data_ = 'prepared/data.csv'
        self.enumerator_ = 'prepared/enumerator.json'
        self.archetype_ = 'prepared/archetype.json'


        '''
        Splitting fractions:
            * 80% of the data for training
            * A quarter, i.e., 0.25/100, of the remaining 20% for testing; 15%
              for validating, 5% for testing.
        '''
        self.fraction = 0.8
        self.aside = 0.25
        self.seed = 5


        '''
        The metadata of the modelling artefacts
        '''
        self.metadata = {'description': 'The modelling artefacts of {architecture}.',
                         'details': 'The {architecture} collection consists of (a) the checkpoints, (b) the logs ' +
                                    'for TensorBoard examination, and (c) much more.'}
