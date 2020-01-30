import numpy as np
from pm.game import Game


class Bandit(Game):

    def __init__(self, X, id=""):
        self._X = X
        self._id = id

    def get_indices(self):
        return self._X

    def get_actions(self, indices):
        return indices

    def get_observation_maps(self, indices):
        return indices.reshape(len(indices), 1, -1)

    def id(self):
        return self._id
