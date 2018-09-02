import pandas as pd
import numpy as np
from .base import Feature
from sklearn.preprocessing import scale
from keras.layers import Input, LSTM, RepeatVector, concatenate, Dense
from keras.models import Model


def split_params(W, U, b, latent_dim):
    Wi = W[:, 0:latent_dim]
    Wf = W[:, latent_dim:2*latent_dim]
    Wc = W[:, 2*latent_dim:3*latent_dim]
    Wo = W[:, 3*latent_dim:]

    Ui = U[:, 0:latent_dim]
    Uf = U[:, latent_dim:2*latent_dim]
    Uc = U[:, 2*latent_dim:3*latent_dim]
    Uo = U[:, 3*latent_dim:]

    bi = b[0:latent_dim]
    bf = b[latent_dim:2*latent_dim]
    bc = b[2*latent_dim:3*latent_dim]
    bo = b[3*latent_dim:]

    return (Wi, Wf, Wc, Wo), (Ui, Uf, Uc, Uo), (bi, bf, bc, bo)


def calc_ht(params):
    x, latent_dim, W_, U_, b_ = params
    Wi, Wf, Wc, Wo = W_
    Ui, Uf, Uc, Uo = U_
    bi, bf, bc, bo = b_
    n = x.shape[0]

    ht_1 = np.zeros(n*latent_dim).reshape(n, latent_dim)
    Ct_1 = np.zeros(n*latent_dim).reshape(n, latent_dim)

    ht_list = []

    for t in np.arange(x.shape[1]):
        xt = np.array(x[:, t, :])
        it = sigmoid(np.dot(xt, Wi) + np.dot(ht_1, Ui) + bi)
        Ct_tilda = np.tanh(np.dot(xt, Wc) + np.dot(ht_1, Uc) + bc)
        ft = sigmoid(np.dot(xt, Wf) + np.dot(ht_1, Uf) + bf)
        Ct = it * Ct_tilda + ft * Ct_1
        ot = sigmoid(np.dot(xt, Wo) + np.dot(ht_1, Uo) + bo)
        ht = ot * np.tanh(Ct)
        ht_list.append(ht)
        ht_1 = ht
        Ct_1 = Ct

    ht = np.array(ht)
    return ht


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


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


class LSTM_giba40cols(Feature):
    def create_features(self, train: pd.DataFrame, test: pd.DataFrame):
        giba40cols = get_selected_features()
        train = train[giba40cols]
        test = test[giba40cols]
        self.suffix = 'giba40cols'

        total = pd.concat([train, test], axis=0)
        total.iloc[:, :] = scale(total)
        train_idx = range(0, len(train))
        test_idx = range(len(train), len(train)+len(test))

        # remove records which is 0 in all columns.
        count_nozero = \
            total.apply(lambda x: (x != 0).sum(), axis=1)
        data = \
            total.iloc[count_nozero[count_nozero != 0].index]
        data = data.values.\
            reshape(-1, total.shape[1], 1)

        # train-test split
        train_test_data = data.copy()
        sampler = np.random.permutation(train_test_data.shape[0])
        train_ratio = 0.8
        train_index_lstm = sampler[:int(train_test_data.shape[0]*train_ratio)]
        valid_index_lstm = sampler[int(train_test_data.shape[0]*train_ratio):]
        x_train = train_test_data[train_index_lstm]
        x_valid = train_test_data[valid_index_lstm]

        # settings
        input_dim = 1
        latent_dim = 5
        timesteps = 40

        # encode
        inputs = Input(shape=(timesteps, input_dim))
        encoded = LSTM(latent_dim,
                       activation="tanh",
                       recurrent_activation="sigmoid",
                       return_sequences=False)(inputs)

        # decode
        hidden = RepeatVector(timesteps)(encoded)
        reverse_input = Input(shape=(timesteps, input_dim))
        hidden_revinput = concatenate([hidden, reverse_input])
        decoded = LSTM(latent_dim, activation="tanh",
                       recurrent_activation="sigmoid",
                       return_sequences=True)(hidden_revinput)
        decoded = Dense(latent_dim, activation="relu")(decoded)
        decoded = Dense(input_dim, activation="tanh")(decoded)

        # train
        LSTM_AE = Model([inputs, reverse_input], decoded)
        LSTM_AE.compile(optimizer='rmsprop', loss='mse')
        x_train_rev = x_train[:, ::-1, :]
        x_valid_rev = x_valid[:, ::-1, :]
        print(x_train_rev.shape)
        LSTM_AE.fit([x_train, x_train_rev], x_train,
                    epochs=30, batch_size=500,
                    shuffle=True,
                    validation_data=([x_valid, x_valid_rev], x_valid))

        # predict
        W, U, b = LSTM_AE.layers[1].get_weights()
        Ws, Us, bs = split_params(W, U, b, latent_dim)

        # for predict
        all_data = total.values.\
            reshape(-1, total.shape[1], 1)

        params = [all_data, latent_dim, Ws, Us, bs]
        ht_data = calc_ht(params)
        ht_columns = [f'lstm{i+1}' for i in range(latent_dim)]
        ht_data = pd.DataFrame(ht_data, columns=ht_columns)

        self.train_feature = ht_data.iloc[train_idx, :]
        self.test_feature = ht_data.iloc[test_idx, :]

        self.train_feature.reset_index(drop=True, inplace=True)
        self.test_feature.reset_index(drop=True, inplace=True)
