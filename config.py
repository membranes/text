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

        The variable <b>self.s3_parameters_template_</b> points to a template of
        Amazon S3 (Simple Storage Service) parameters & arguments.
        """

        self.warehouse = os.path.join(os.getcwd(), 'warehouse')
        self.artefacts_ = os.path.join(self.warehouse, 'artefacts')
        self.s3_parameters_template_ = 'https://raw.githubusercontent.com/membranes/configurations/refs/heads/master/data/s3_parameters.yaml'


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
            * A tenth of the remaining 20% for testing, i.e., 18%
              for validating, 2% for testing
        '''
        self.fraction = 0.8
        self.aside = 0.1
        self.seed = 5
