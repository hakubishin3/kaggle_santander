import pandas as pd
import numpy as np
import umap
from .base import Feature


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


def build_histograms(data: pd.DataFrame):
    df_X = (data.replace(0, np.nan).apply(np.log) * 10).round()
    start = int(df_X.min().min())
    stop = int(df_X.max().max())
    return pd.DataFrame(
        data={f'bucket{cnt}': (df_X == cnt).sum(axis=1) for cnt in range(start, stop+1)})


class BuildHist(Feature):
    def create_features(self, train: pd.DataFrame, test: pd.DataFrame,
                        n_comp=3, random_state=71):
        total = pd.concat([train, test], axis=0)
        train_idx = range(0, len(train))
        test_idx = range(len(train), len(train)+len(test))
        build_results = build_histograms(total)

        umap_ = umap.UMAP(n_components=n_comp, random_state=random_state)
        umap_results = umap_.fit_transform(build_results)

        for i in range(0, umap_results.shape[1]):
            self.train_feature[f'build_umap{i+1}'] = umap_results[train_idx, i]
            self.test_feature[f'build_umap{i+1}'] = umap_results[test_idx, i]

        self.train_feature.reset_index(drop=True, inplace=True)
        self.test_feature.reset_index(drop=True, inplace=True)


class BuildHist_giba40cols(Feature):
    def create_features(self, train: pd.DataFrame, test: pd.DataFrame,
                        n_comp=3, random_state=71):
        giba40cols = get_selected_features()
        train = train[giba40cols]
        test = test[giba40cols]
        self.suffix = 'giba40cols'

        total = pd.concat([train, test], axis=0)
        train_idx = range(0, len(train))
        test_idx = range(len(train), len(train)+len(test))
        build_results = build_histograms(total)

        umap_ = umap.UMAP(n_components=n_comp, random_state=random_state)
        umap_results = umap_.fit_transform(build_results)

        for i in range(0, umap_results.shape[1]):
            self.train_feature[f'build_umap{i+1}'] = umap_results[train_idx, i]
            self.test_feature[f'build_umap{i+1}'] = umap_results[test_idx, i]

        self.train_feature.reset_index(drop=True, inplace=True)
        self.test_feature.reset_index(drop=True, inplace=True)
