"""Module tags.py"""
import logging
import pandas as pd


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

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger: logging.Logger = logging.getLogger(name=__name__)

    def exc(self):

        elements: pd.DataFrame = self.__tag_data.groupby(by=self.__tag_fields).value_counts().to_frame()
        elements.reset_index(drop=False, inplace=True)

        self.__logger.info(msg=elements)
