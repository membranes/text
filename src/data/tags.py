"""Module tags.py"""
import logging
import numpy as np
import pandas as pd

import config

class Tags:
    """
    Tags
    ----

    Examine data balance/imbalance by tag frequency.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Constructor
        """

        self.__tag_fields: list[str] = ['tag', 'annotation', 'category']
        self.__tag_data: pd.DataFrame = data.copy()[self.__tag_fields]

        # Categories
        self.__mcf: int = config.Config().minimum_category_frequency

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger: logging.Logger = logging.getLogger(name=__name__)

    def __categories(self, blob: pd.DataFrame) -> np.ndarray:
        """
        
        :param blob: A frame that includes {category}, and {count} per category
        """

        categories: pd.DataFrame = blob[['category', 'count']].groupby(by='category').sum().reset_index(drop=False)
        self.__logger.info(msg=categories)

        categories = categories.copy().loc[categories['count'] >= self.__mcf, :]
        self.__logger.info(msg=categories)

        return categories['category'].values

    def exc(self):

        elements: pd.DataFrame = self.__tag_data.groupby(by=self.__tag_fields).value_counts().to_frame()
        elements.reset_index(drop=False, inplace=True)

        categories = self.__categories(blob=elements)

        self.__logger.info(msg=elements)
