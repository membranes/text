"""Module main.py"""
import argparse
import logging
import os
import sys

import pandas as pd
import ray
import torch


def main():
    """
    Entry Point

    :return:
    """

    logger: logging.Logger = logging.getLogger(__name__)

    # Set up
    setup: bool = src.setup.Setup(service=service, s3_parameters=s3_parameters, architecture=args.architecture).exc()
    if not setup:
        src.functions.cache.Cache().exc()
        sys.exit('No Executions')

    # Device Selection: Setting a graphics processing unit as the default device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info('Device: %s', device)

    # Ray
    ray.init(dashboard_host='172.17.0.2', dashboard_port=8265)

    # The Data
    interface = src.data.interface.Interface(s3_parameters=s3_parameters).data()
    data: pd.DataFrame = interface.data()

    # Temporary
    data = data.loc[:500, :]
    src.models.interface.Interface(
        data=data, enumerator=interface.enumerator(), archetype=interface.archetype()).exc(
        architecture=args.architecture, arguments=arguments, hyperspace=hyperspace)

    src.data.transfer.Transfer(
        service=service, s3_parameters=s3_parameters, architecture=args.architecture).exc()

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
    import src.data.interface
    import src.data.transfer
    import src.elements.arguments
    import src.functions.cache
    import src.functions.expecting
    import src.functions.service
    import src.models.interface
    import src.models.arguments
    import src.models.hyperspace
    import src.s3.s3_parameters
    import src.setup

    expecting = src.functions.expecting.Expecting()
    parser = argparse.ArgumentParser()
    parser.add_argument('architecture', type=expecting.architecture,
                        help='The name of the architecture in focus.')
    args = parser.parse_args()

    # S3 S3Parameters, Service Instance
    s3_parameters = src.s3.s3_parameters.S3Parameters().exc()
    service = src.functions.service.Service(region_name=s3_parameters.region_name).exc()

    arguments = src.models.arguments.Arguments(s3_parameters=s3_parameters).exc(
        node=f'{args.architecture}/arguments.json')

    hyperspace = src.models.hyperspace.Hyperspace(service=service, s3_parameters=s3_parameters).exc(
        node=f'{args.architecture}/hyperspace.json'
    )

    main()
