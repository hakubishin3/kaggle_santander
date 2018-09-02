from abc import ABCMeta, abstractmethod


class Model(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, x_train, y_train, x_valid, y_valid, params):
        raise NotImplementedError
