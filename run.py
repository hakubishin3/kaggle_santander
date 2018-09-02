import json
import argparse
import logging
import numpy as np
from src.data.load_dataset import get_dataset_filename, load_dataset
from src.utils.logger_functions import get_module_logger
from src.features.aggregates import Aggregates


feature_map = {
    'Aggregates': Aggregates
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/lightgbm_1.json')
    parser.add_argument('--debug_mode', '-d', action='store_true')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Overwrite existing files')
    args = parser.parse_args()
    import pdb; pdb.set_trace()

    logger = get_module_logger(__name__)
    config = json.load(open(args.config))

    # load dataset
    logger.info('load dataset')
    train_path = get_dataset_filename(config, 'train')
    test_path = get_dataset_filename(config, 'test')
    train, test = load_dataset(
        train_path, test_path, args.debug_mode)
    logger.debug(f'train: {train.shape}, test: {test.shape}')

    y_train = np.log1p(train['target'].values)
    test_index = test['ID'].values
    train.drop(['ID', 'target'], axis=1, inplace=True)
    test.drop(['ID'], axis=1, inplace=True)

    # make features
    make_features(train, test, config, args.force)
    x_train, x_test = load_features(config)

    # train model


if __name__ == '__main__':
    main()
