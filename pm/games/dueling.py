import numpy as np
from pm.game import Game


class DuelingBandit(Game):

    def __init__(self):
        X0 = []
        self._index_set = cartesian(x0)

    def get_indices(self):
        return self._index_set

    def get_actions(self, indices):
        return (indices[:, 0] + indices[:, 1])/2

    def get_observation_maps(self, indices):
        return indices[:, 0] - indices[:, 1]

