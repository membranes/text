import os
import pandas as pd
import dask.dataframe as dfr

import config

class Fundamentals:

    def __init__(self) -> None:
        
        self.__datapath = config.Config().datapath 

        self.__names: dict[str, str] = {'sentence #': 'sentence_identifier', 'pos': 'part'}

    def __read(self) -> pd.DataFrame:

        try:
             frame: pd.DataFrame = pd.read_csv(filepath_or_buffer=os.path.join(self.__datapath, 'dataset.csv'), header=0, encoding='utf-8')
        except ImportError as err:
            raise err from err
        
        frame.loc[:, 'Sentence #'] = frame['Sentence #'].ffill().values

        print(type(frame))
        
        return frame

    def __rename(self, blob: pd.DataFrame) -> pd.DataFrame:

        frame: pd.DataFrame = blob.copy()
        frame.rename(mapper=str.lower, axis=1, inplace=True)
        frame.rename(columns={'sentence #': 'sentence_identifier', 'pos': 'part'}, inplace=True)
        
        return frame

    def exc(self) -> pd.DataFrame:

        frame: pd.DataFrame = self.__read()
        frame = self.__rename(blob=frame)

        return frame
        
        

