import numpy as np
from pm.game import Game


class Bandit(Game):

    def __init__(self, X, id=""):
        self._X = X
        self._id = id
        self._d = X.shape[1]
        self._I = np.arange(len(X))

    def get_d(self):
        return self._d

    def get_indices(self):
        return self._I

    def get_actions(self, indices):
        return self._X[indices]

    def get_observation_maps(self, indices):
        return self.get_actions(indices).reshape(len(indices), 1, self._d)

    def id(self):
        return self._id
