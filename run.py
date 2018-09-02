import json
import argparse
import logging
import numpy as np
from src.data.load_dataset import get_dataset_filename, load_dataset
from src.utils.logger_functions import get_module_logger
from src.features.base import load_features
from src.features.Aggregates import Aggregates, Aggregates_giba40cols
from src.features.Decomposition import Decomposition, Decomposition_giba40cols
from src.features.BuildHist import BuildHist, BuildHist_giba40cols
from src.features.LSTM import LSTM_giba40cols

feature_map = {
    'Aggregates': Aggregates,
    'Aggregates_giba40cols': Aggregates_giba40cols,
    'Decomposition': Decomposition,
    'Decomposition_giba40cols': Decomposition_giba40cols,
    "BuildHist": BuildHist,
    "BuildHist_giba40cols": BuildHist_giba40cols,
    "LSTM_giba40cols": LSTM_giba40cols
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/lightgbm_1.json')
    parser.add_argument('--debug_mode', '-d', action='store_true')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Overwrite existing files')
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

    y_train = np.log1p(train['target'].values)
    test_index = test['ID'].values
    train.drop(['ID', 'target'], axis=1, inplace=True)
    test.drop(['ID'], axis=1, inplace=True)

    # make features
    logger.info('make features')
    feature_path = config['dataset']['feature_directory']
    target_feature_map = \
        {name: key for name, key in feature_map.items() if name in config['features']}

    for name, key in target_feature_map.items():
            f = key(feature_path)
            if f.train_path.exists() and f.test_path.exists() and not args.force:
                logger.info(f'{f.name} was skipped')
            else:
                f.run(train, test).save()

    # load features
    logger.info('load features')
    x_train, x_test = load_features(config)
    logger.debug(f'number of features: {x_train.shape[1]}')

    # train model


if __name__ == '__main__':
    main()
