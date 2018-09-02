import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from .base import Model


class LightGBM(object):
    def fit(self, d_train: lgb.Dataset, d_valid: lgb.Dataset, params: dict):
        evals_result = {}
        model = lgb.train(
            params=params['model_params'],
            train_set=d_train,
            valid_sets=[d_train, d_valid],
            valid_names=['train', 'valid'],
            evals_result=evals_result,
            **params['train_params']
            )
        return model, evals_result

    def cv(self, x_train, y_train, x_test, params, n_folds=5):
        # create folds
        folds = KFold(n_splits=5, shuffle=True, random_state=71)
        folds.get_n_splits(x_train)

        # make Dataset object
        d_train = lgb.Dataset(data=x_train, label=y_train, free_raw_data=True)
        d_train.construct()

        # init predictions
        sub_preds = np.zeros(x_test.shape[0])
        oof_preds = np.zeros(x_train.shape[0])

        # Run Kfolds
        for trn_idx, val_idx in folds.split(x_train):
            # train model
            model, evals_result = self.fit(
                d_train.subset(trn_idx), d_train.subset(val_idx), params
                )

            # predict out-of-fold and test
            oof_preds[val_idx] = model.predict(x_train.iloc[val_idx])
            sub_preds += model.predict(x_test) / n_folds

        # display current out-of-fold score
        oof_score = mean_squared_error(y_train, oof_preds)**0.5

        return oof_preds, sub_preds
