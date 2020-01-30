import numpy as np
from scipy.linalg import cho_factor, cho_solve

from pm.utils import difference_matrix


class RegularizedLeastSquares:

    def __init__(self, d):
        self._d = d
        self._V = np.eye(self._d)
        self._XY = np.zeros(self._d)
        self._update_cache()

    def _update_cache(self):
        self._cholesky = cho_factor(self._V)
        self._theta = cho_solve(self._cholesky, self._XY)

    def add_data(self, x, y):
        self._V += x.T.dot(x)
        self._XY += x.T.dot(y)
        self._update_cache()

    def theta(self):
        return self._theta

    def mean(self, x):
        return x.dot(self._theta)

    def var(self, x):
        """
        Computes variance at x
        :param x:
        :return: x^T V_t^{-1} x
        """
        sol_x = cho_solve(self._cholesky, x.T).T
        return np.sum(x*sol_x, axis=-1)

    def ucb(self, x, delta):
        """
        Upper confidence bound for x at confidence level delta
        """
        return self.mean(x) + self.beta(delta)*np.sqrt(self.var(x))

    def lcb(self, x, delta):
        """
        Upper confidence bound for x at confidence level delta
        """
        return self.mean(x) - self.beta(delta) * np.sqrt(self.var(x))

    def beta(self, delta):
        """
        compute beta = sqrt(log det(V_t) + 2 log(1/delta)) + 1
        :param delta:
        :return:
        """
        logdet = 2 * np.sum(np.log(np.diag(self._cholesky[0])))
        beta = np.sqrt(logdet - 2*np.log(delta)) + 1
        return beta

    def get_cholesky_factor(self):
        return self._cholesky

    def get_V(self):
        return self._V


class RegretEstimator:

    def __init__(self, game, lls, delta, truncate=True):
        self._truncate = truncate
        self._game = game
        self._d = self._game.get_d()
        self._lls = lls
        self._delta = delta

    @property
    def lls(self):
        return self._lls

    def add_data(self, indices, y):
        # stack the observation operators for the given indices

        # TODO: The atleast_2d is not very nice, there should be a specified format
        ax = np.atleast_2d(self._game.get_observation_maps(indices))
        if ax.shape[0] > 1:
            ax = np.hstack(ax)
        y = np.atleast_1d(y)
        self._lls.add_data(ax, y)

    def ucb(self, indices):
        X = self._game.get_actions(indices)

        if self._truncate:
            return np.minimum(self._lls.ucb(X, delta=self._delta), 1)

        return self._lls.ucb(X, delta=self._delta)

    def regret_upper(self, indices):
        """
        Regret upper bound
        """
        X = self._game.get_actions(indices)
        D = difference_matrix(X)

        # compute ucb score for all differences and max out rows
        regret = np.max(self._lls.ucb(D, delta=self._delta), axis=-1)

        if self._truncate:
            return np.minimum(regret, 1)

        return regret

    def regret_lower_1(self, indices):
        raise NotImplemented

    def regret_lower_2(self, indices):
        """
        Relaxed regret lower bound
        """
        X = self._game.get_actions(indices)
        D = difference_matrix(X)

        # compute ucb score for all differences and max out rows
        regret = np.max(self._lls.lcb(D, delta=self._delta), axis=-1)

        return regret



