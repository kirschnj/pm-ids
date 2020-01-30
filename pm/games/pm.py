import numpy as np

from pm.game import Game, GameInstance


class GenericPM(Game):
    """
    Generic Partial Monitoring Game
    """

    def __init__(self, X, A, id=""):

        if np.ndim(X) != 2:
            raise ValueError("X needs to be 2-dimensional!")

        if np.ndim(A) != 3:
            raise ValueError("A needs to be 3-dimensional!")

        if X.shape[1] != A.shape[2]:
            raise ValueError("Last dimension of X and A need to be the same!")

        if len(X) != len(A):
            raise ValueError("X and A must have the same length, len(X) == len(A)!")

        self._I = np.arange(len(X))
        self._X = X
        self._A = A
        self._id = id

    def get_indices(self):
        return self._I

    def get_actions(self, indices):
        return self._X[indices]

    def get_observation_maps(self, indices):
        return self._A[indices]

    def id(self):
        return self._id


