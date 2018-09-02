import umap
import pandas as pd
from .base import Feature
from sklearn.decomposition import \
    PCA, TruncatedSVD, FastICA, FactorAnalysis, KernelPCA
from sklearn.random_projection import \
    GaussianRandomProjection, SparseRandomProjection


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


def get_decomp_dict(n_comp, random_state):
    pca = PCA(n_components=n_comp, random_state=random_state)
    tsvd = TruncatedSVD(n_components=n_comp, random_state=random_state)
    fact = FactorAnalysis(n_components=n_comp, random_state=random_state)
    ica = FastICA(n_components=n_comp, random_state=random_state)
    grp = GaussianRandomProjection(n_components=n_comp, eps=0.1,
                                   random_state=random_state)
    srp = SparseRandomProjection(n_components=n_comp, dense_output=True,
                                 random_state=random_state)
    umap_ = umap.UMAP(n_components=n_comp, random_state=random_state)

    decomp_method_dict = {
        "pca": pca, "tsvd": tsvd, "fact": fact,
        "ica": ica, "grp": grp, "srp": srp, "umap": umap_
        }

    return decomp_method_dict


class Decomposition(Feature):
    def create_features(self, train: pd.DataFrame, test: pd.DataFrame,
                        n_comp=3, random_state=71):
        decomp_method_dict = get_decomp_dict(n_comp, random_state)

        for key, decomp in decomp_method_dict.items():
            decomp_train = decomp.fit_transform(train)
            decomp_test = decomp.transform(test)

            for i in range(0, decomp_train.shape[1]):
                self.train_feature[f'{key}{i+1}'] = decomp_train[:, i]
                self.test_feature[f'{key}{i+1}'] = decomp_test[:, i]

        self.train_feature.reset_index(drop=True, inplace=True)
        self.test_feature.reset_index(drop=True, inplace=True)


class Decomposition_giba40cols(Feature):
    def create_features(self, train: pd.DataFrame, test: pd.DataFrame,
                        n_comp=3, random_state=71):
        giba40cols = get_selected_features()
        train = train[giba40cols]
        test = test[giba40cols]
        self.suffix = 'giba40cols'

        decomp_method_dict = get_decomp_dict(n_comp, random_state)

        for key, decomp in decomp_method_dict.items():
            decomp_train = decomp.fit_transform(train)
            decomp_test = decomp.transform(test)

            for i in range(0, decomp_train.shape[1]):
                self.train_feature[f'{key}{i+1}'] = decomp_train[:, i]
                self.test_feature[f'{key}{i+1}'] = decomp_test[:, i]

        self.train_feature.reset_index(drop=True, inplace=True)
        self.test_feature.reset_index(drop=True, inplace=True)
