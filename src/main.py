import logging
import os
import sys

import pandas as pd
import torch
import ray


def main():
    """
    https://docs.ray.io/en/latest/ray-core/api/doc/ray.init.html
    https://docs.ray.io/en/latest/ray-core/configure.html

    :return:
    """

    logger: logging.Logger = logging.getLogger(__name__)

    # Set up
    setup: bool = src.setup.Setup(service=service, s3_parameters=s3_parameters).exc()

    # Device Selection: Setting a graphics processing unit as the default device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info('Device: %s', device)

    # Ray
    ray.init(dashboard_host='172.17.0.2', dashboard_port=8265)

    # The Data
    # data: pd.DataFrame = src.data.source.Source().exc()
    if setup:
        interface = src.data.interface.Interface(s3_parameters=s3_parameters)
        data = interface.data()
        logger.info(data.head())
        data.info()
    
    # Tags
    # elements, enumerator, archetype = src.data.tags.Tags(data=data).exc()
    # logger.info('Enumerator: %s', enumerator)
    # logger.info('Archetype: %s', archetype)
    
    # Balance/Imbalance
    # data = data.copy().loc[data['category'].isin(values=elements['category'].unique()), :]

    # Sentences & Labels
    # frame: pd.DataFrame = src.data.demarcations.Demarcations(data=data).exc()
    # logger.info('Data:\n%s', frame.head())

    # Temporary
    # frame = frame.loc[:500, :]
    # src.models.interface.Interface(frame=frame, enumerator=enumerator, archetype=archetype).exc()

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

    # Activate graphics processing units
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    os.environ['TOKENIZERS_PARALLELISM']='true'
    os.environ['RAY_USAGE_STATS_ENABLED']='0'
    
    # Modules
    import src.data.source
    import src.data.tags
    import src.data.demarcations
    import src.data.interface

    import src.elements.s3_parameters as s3p
    import src.elements.service as sr

    import src.functions.cache
    import src.functions.service

    import src.models.interface
    import src.s3.s3_parameters
    import src.setup


    # S3 S3Parameters, Service Instance
    s3_parameters: s3p.S3Parameters = src.s3.s3_parameters.S3Parameters().exc()
    service: sr.Service = src.functions.service.Service(region_name=s3_parameters.region_name).exc()

    main()
