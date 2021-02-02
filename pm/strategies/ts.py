from scipy.linalg import solve_triangular

from pm.strategy import Strategy
import numpy as np

class TS(Strategy):
    def __init__(self, game, estimator, noise_std=1.):
        super().__init__(game, estimator)
        self._noise_std = noise_std

    def get_next_action(self):
        lls = self._estimator._lls
        eta = np.random.normal(0, 1, self._game.d)
        # print(eta)
        # print(type(eta))
        # print(lls._cholesky)
        # print(lls._cholesky.dtype)
        sample = lls.theta + self._noise_std*solve_triangular(lls._cholesky[0], eta)
        indices = self._game.get_indices()
        X = self._game.get_actions(indices)
        reward = X.dot(sample)
        return indices[np.argmax(reward)]
