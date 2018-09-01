import json
import argparse
import logging
from src.data.load_dataset import get_dataset_filename, load_dataset
from src.utils.logger_functions import get_module_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/lightgbm_1.json')
    parser.add_argument('--debug_mode', action='store_true')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    config = json.load(open(args.config))

    # load dataset
    logger.info('load dataset')
    train_path = get_dataset_filename(config, 'train')
    test_path = get_dataset_filename(config, 'test')
    train, test = load_dataset(
        train_path, test_path, args.debug_mode)
    logger.debug(f'train: {train.shape}, test: {test.shape}')

    # make features
    



if __name__ == '__main__':
    main()
