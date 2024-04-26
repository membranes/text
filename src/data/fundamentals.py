import os
import pandas as pd
import dask.dataframe as dfr
import dask_expr._collection

import config

class Fundamentals:

    def __init__(self) -> None:
        
        self.__datapath = config.Config().datapath 

        self.__names: dict[str, str] = {'Word': 'word', 'POS': 'part', 'Tag': 'tag'}

    def __read(self) -> dfr.DataFrame:

        try:
            frame: dfr.DataFrame = dfr.read_csv(path=os.path.join(self.__datapath, 'dataset.csv'), header=0)
        except ImportError as err:
            raise err from err
        
        frame: dfr.DataFrame = frame.assign(sentence_identifier=frame['Sentence #'].ffill())
        frame: dfr.DataFrame = frame.drop(columns='Sentence #')
                
        return frame

    def __rename(self, blob: dfr.DataFrame) -> dfr.DataFrame:

        frame: dfr.DataFrame = blob.copy()
        frame: dfr.DataFrame = frame.rename(columns=self.__names)
        
        return frame
    
    def __tag_splits(self, blob: dfr.DataFrame) -> dfr.DataFrame:

        splits: dfr.DataFrame = blob['tag'].str.split(pat='-', n=2, expand=True)
        splits: dfr.DataFrame = splits.rename(columns={0: 'annotation', 1: 'category'})
        splits: dfr.DataFrame = splits.assign(category=splits['category'].fillna(value='O'))
        frame : dfr.DataFrame = blob.join(other=splits)
        
        return frame

    def exc(self) -> pd.DataFrame:

        frame: dfr.DataFrame = self.__read()
        frame: dfr.DataFrame = self.__rename(blob=frame)
        frame = self.__tag_splits(blob=frame)
        
        return frame.compute()
