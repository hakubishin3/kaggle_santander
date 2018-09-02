import re
import time
import pandas as pd
from pathlib import Path
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager


@contextmanager
def timer(name):
    t0 = time.time()
    print(f'[{name}] start')
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


class Feature(metaclass=ABCMeta):
    prefix = ''
    suffix = ''
    dir = '.'

    def __init__(self):
        self.name = self.__class__.__name__
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.train_path = Path(self.dir) / f'{self.name}_train.ftr'
        self.test_path = Path(self.dir) / f'{self.name}_test.ftr'

    def run(self):
        with timer(self.name):
            self.create_features()
            prefix = self.prefix + '_' if self.prefix else ''
            suffix = '_' + self.suffix if self.suffix else ''
            self.train.columns = prefix + self.train.columns + suffix
            self.test.columns = prefix + self.test.columns + suffix
        return self

    @abstractmethod
    def create_features(self):
        raise NotImplementedError

    def save(self):
        self.train.to_feather(str(self.train_path))
        self.test.to_feather(str(self.test_path))


def make_features(train, test, config, overwrite: bool):
    for f in config['features']:
        print("a")


def load_features(config):
    feathre_path = config['dataset']['feature_directory']

    dfs = [pd.read_feather(
        f'{feathre_path}/{f}_train.ftr') for f in config['features']]
    x_train = pd.concat(dfs, axis=1)

    dfs = [pd.read_feather(
        f'{feathre_path}/{f}_test.ftr') for f in config['features']]
    x_test = pd.concat(dfs, axis=1)

    return x_train, x_test
