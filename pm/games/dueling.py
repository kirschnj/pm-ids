from itertools import product

import numpy as np
from pm.game import Game


class DuelingBandit(Game):

    def __init__(self, X_base, name=None):
        X = np.empty(shape=(X_base.shape[0]**2, X_base.shape[1]))
        M = np.empty(shape=(X_base.shape[0]**2, X_base.shape[1]))

        for i, (x1, x2) in enumerate(product(X_base, X_base)):
            X[i] = x1 + x2
            M[i] = x1 - x2

        self.X_base = X_base
        super().__init__(X,M, name)

    def get_base_actions(self):
        return self.X_base


class LocalizedDuelingBandit(Game):
    def __init__(self, X_base, name=None):
        d = X_base.shape[1]
        X = np.empty(shape=(2*X_base.shape[0] - 1, d))
        M = np.empty(shape=(2*X_base.shape[0] - 1, d))

        for i, (x1, x2) in enumerate(zip(X_base[:-1], X_base[1:])):
            X[i] = x1 + x2
            M[i] = x1 - x2

        for i, x1 in enumerate(X_base):
            X[i+d-1] = 2*x1
            M[i+d-1] = np.zeros_like(x1)

        super().__init__(X, M, name)

