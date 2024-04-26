"""Module tags.py"""
import logging
import pandas as pd


class Tags:
    """
    Tags
    ----

    Examine data balance/imbalance by tag frequency.
    """

    def __init__(self) -> None:
        """
        Constructor
        """
        
        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger: logging.Logger = logging.getLogger(name=__name__)

    def exc(self, series: pd.Series):

        elements: pd.DataFrame = series.value_counts().to_frame().reset_index(drop=False)

        self.__logger.info(elements)
