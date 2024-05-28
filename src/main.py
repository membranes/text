import logging
import os
import sys

import pandas as pd
import torch

def main():

    logger: logging.Logger = logging.getLogger(__name__)

    # Device Selection: Setting a graphics processing unit as the default device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(msg=device)

    # The Data
    data: pd.DataFrame = src.data.source.Source().exc()
    logger.info(data.head())
    
    # Tags
    elements, enumerator, denumerator = src.data.tags.Tags(data=data).exc()
    
    # Balance/Imbalance
    data = data.copy().loc[data['category'].isin(values=elements['category'].unique()), :]

    # Sentences & Labels
    frame: pd.DataFrame = src.data.demarcations.Dermacations(data=data).exc()
    logger.info(frame)

    src.structures.initial.Initial(frame=frame, enumerator=enumerator).exc()

    # Delete Cache Points
    src.functions.cache.Cache().exc()


if __name__ == '__main__':

    # Paths
    root = os.getcwd()
    sys.path.append(root)
    sys.path.append(os.path.join(root, 'src'))

    # Logging
    logging.basicConfig(level=logging.INFO,
                        format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                        datefmt='%Y-%m-%d %H:%M:%S')
    
    # Modules
    import src.data.source
    import src.data.tags
    import src.data.demarcations
    import src.functions.cache
    import src.structures.initial
    
    main()
