import pandas as pd
import numpy as np
from .base import Feature
from scipy.stats import skew, kurtosis


def get_selected_features():
    return [
        'f190486d6', 'c47340d97', 'eeb9cd3aa', '66ace2992', 'e176a204a',
        '491b9ee45', '1db387535', 'c5a231d81', '0572565c2', '024c577b9',
        '15ace8c9f', '23310aa6f', '9fd594eec', '58e2e02e6', '91f701ba2',
        'adb64ff71', '2ec5b290f', '703885424', '26fc93eb7', '6619d81fc',
        '0ff32eb98', '70feb1494', '58e056e12', '1931ccfdd', '1702b5bf0',
        '58232a6fb', '963a49cdc', 'fc99f9426', '241f0f867', '5c6487af1',
        '62e59a501', 'f74e8f13d', 'fb49e4212', '190db8488', '324921c7b',
        'b43a7cfd5', '9306da53f', 'd6bb78916', 'fb0f5dbfe', '6eef030c1'
    ]


class Aggregates(Feature):
    def create_features(self, train: pd.DataFrame, test: pd.DataFrame):
        stats = [np.min, np.max, np.mean, np.median,
                 np.std, np.sum, skew, kurtosis
                 ]

        for stat in stats:
            stat_name = stat.__name__
            self.train_feature[stat_name] = train.apply(
                lambda x: stat(x[x != 0]), axis=1)
            self.test_feature[stat_name] = test.apply(
                lambda x: stat(x[x != 0]), axis=1)

        self.train_feature['number_of_different'] = train.nunique(axis=1)
        self.train_feature['non_zero_count'] = train.astype(bool).sum(axis=1)
        self.train_feature[f'geometric_mean'] = train.apply(
            lambda x: np.exp(np.log(x[x > 0]).mean()), axis=1)

        self.test_feature['number_of_different'] = test.nunique(axis=1)
        self.test_feature['non_zero_count'] = test.astype(bool).sum(axis=1)
        self.test_feature[f'geometric_mean'] = test.apply(
            lambda x: np.exp(np.log(x[x > 0]).mean()), axis=1)

        self.train_feature.fillna(value=0, inplace=True)
        self.test_feature.fillna(value=0, inplace=True)

        self.train_feature.reset_index(drop=True, inplace=True)
        self.test_feature.reset_index(drop=True, inplace=True)


class Aggregates_giba40cols(Feature):
    def create_features(self, train: pd.DataFrame, test: pd.DataFrame):
        giba40cols = get_selected_features()
        train = train[giba40cols]
        test = test[giba40cols]
        self.suffix = 'giba40cols'

        stats = [np.min, np.max, np.mean, np.median,
                 np.std, np.sum, skew, kurtosis
                 ]

        for stat in stats:
            stat_name = stat.__name__
            self.train_feature[stat_name] = train.apply(
                lambda x: stat(x[x != 0]), axis=1)
            self.test_feature[stat_name] = test.apply(
                lambda x: stat(x[x != 0]), axis=1)

        self.train_feature['number_of_different'] = train.nunique(axis=1)
        self.train_feature['non_zero_count'] = train.astype(bool).sum(axis=1)
        self.train_feature[f'geometric_mean'] = train.apply(
            lambda x: np.exp(np.log(x[x > 0]).mean()), axis=1)

        self.test_feature['number_of_different'] = test.nunique(axis=1)
        self.test_feature['non_zero_count'] = test.astype(bool).sum(axis=1)
        self.test_feature[f'geometric_mean'] = test.apply(
            lambda x: np.exp(np.log(x[x > 0]).mean()), axis=1)

        self.train_feature.fillna(value=0, inplace=True)
        self.test_feature.fillna(value=0, inplace=True)

        self.train_feature.reset_index(drop=True, inplace=True)
        self.test_feature.reset_index(drop=True, inplace=True)
