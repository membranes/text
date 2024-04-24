import os
import pandas as pd
import dask.dataframe as dfr
import dask_expr._collection

import config

class Fundamentals:

    def __init__(self) -> None:
        
        self.__datapath = config.Config().datapath 

        self.__names: dict[str, str] = {'Word': 'word',
                                        'POS': 'part', 'Tag': 'tag'}

    def __read(self) -> dfr.DataFrame:

        try:
            frame: dfr.DataFrame = dfr.read_csv(path=os.path.join(self.__datapath, 'dataset.csv'), header=0)
            # frame: pd.DataFrame = pd.read_csv(filepath_or_buffer=os.path.join(self.__datapath, 'dataset.csv'), header=0, encoding='utf-8')
        except ImportError as err:
            raise err from err
        
        frame = frame.assign(sentence_identifier=frame['Sentence #'].ffill())
        frame = frame.drop(columns='Sentence #')
                
        return frame

    def __rename(self, blob):

        frame = blob.copy()
        # frame.rename(mapper=str.lower, axis=1, inplace=True)
        frame = frame.rename(columns=self.__names)
        
        return frame

    def exc(self) -> pd.DataFrame:

        frame = self.__read()
        frame = self.__rename(blob=frame)
        
        return frame.compute()
