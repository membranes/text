import logging

import pandas as pd


class Dermacations:

    def __init__(self, data: pd.DataFrame) -> None:
        
        self.__data: pd.DataFrame = data

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def __call__(self) -> pd.DataFrame:

        T = self.__data[['sentence_identifier', 'word', 'tag']].copy()
        T.info()

        sentences: pd.DataFrame = T.drop(columns='tag').groupby(
            by=['sentence_identifier'])['word'].apply(lambda x: ' '.join(x)).to_frame()

        labels: pd.DataFrame = T.drop(columns='word').groupby(
            by=['sentence_identifier'])['tag'].apply(lambda x: ','.join(x)).to_frame()
        
        
        frame: pd.DataFrame = sentences.join(labels).drop_duplicates()
        frame.reset_index(inplace=True)

        return frame
