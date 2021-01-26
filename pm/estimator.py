import numpy as np
from scipy.linalg import cho_factor, cho_solve

from pm.utils import difference_matrix


class RegularizedLeastSquares:

    def __init__(self, d, beta_logdet=True, noise_var=1.):
        self._d = d
        self._V = np.eye(self._d)
        self._XY = np.zeros(self._d)
        self._s = 1
        self.beta_logdet = beta_logdet
        self.noise_var = noise_var
        self.noise_std = np.sqrt(self.noise_var)

        self._update_cache()

    def _update_cache(self):
        self._cholesky = cho_factor(self._V)
        self._theta = cho_solve(self._cholesky, self._XY)

    def add_data(self, x, y):
        self._V += x.T.dot(x)
        self._XY += x.T.dot(y)
        self._update_cache()
        self._s += len(y)

    @property
    def theta(self):
        """ estimated parameter """
        return self._theta

    def mean(self, x):
        return x.dot(self._theta)

    @property
    def s(self):
        """ number of data points added to the estimator """
        return self._s

    def var(self, x):
        """
        Computes variance at x
        :param x:
        :return: x^T V_t^{-1} x
        """
        sol_x = cho_solve(self._cholesky, x.T).T
        return np.sum(x*sol_x, axis=-1)

    def ucb(self, x, delta=None):
        """
        Upper confidence bound for x at confidence level delta
        """
        return self.mean(x) + np.sqrt(self.beta(delta) * self.var(x))

    def lcb(self, x, delta=None):
        """
        Upper confidence bound for x at confidence level delta
        """
        return self.mean(x) - np.sqrt(self.beta(delta) * self.var(x))

    def beta(self, delta=None):
        """
        compute beta = sqrt(log det(V_t) + 2 log(1/delta)) + 1
        :param delta:
        :return:
        """
        _s = max(self.s, 2)
        if delta is None:
            delta = 1/(_s*np.log(_s))
        if self.beta_logdet:
            logdet = 2 * np.sum(np.log(np.diag(self._cholesky[0])))
            beta = (self.noise_std*np.sqrt(logdet + 2*np.log(1/delta)) + 1)**2
        else:
            beta = self.noise_var*2*np.log(1/delta) + self._d * max(np.log(np.log(_s)), 1)
        return beta

    def get_cholesky_factor(self):
        return self._cholesky

    @property
    def V(self):
        """ co-variance matrix of the estimator """
        return self._V


class RegretEstimator:

    def __init__(self, game, lls, truncate=True, delta=None, ucb_estimates=True):
        self._truncate = truncate
        self._game = game
        self._d = self._game.get_d()
        self._lls = lls
        self._delta = delta
        self._ucb_estimates = ucb_estimates  # this flag determines the way the gap upper bound is computed

    @property
    def lls(self):
        return self._lls

    def add_data(self, indices, y):
        # stack the observations and observation operators
        ax = self._game.get_observation_maps(indices)
        ax = np.vstack(ax)
        y = np.hstack(y)

        # update lls
        self._lls.add_data(ax, y)

    def ucb(self, indices, delta=None):
        X = self._game.get_actions(indices)

        # if self._truncate:
        #     return np.minimum(self._lls.ucb(X, delta=self._delta), 1)
        if delta == None:
            return self._lls.ucb(X, delta=self._delta)
        else:
            return self._lls.ucb(X, delta=delta)

    def lcb(self, indices,):
        X = self._game.get_actions(indices)
        return self._lls.lcb(X, delta=self._delta)

    def var(self, indices):
        X = self._game.get_actions(indices)
        return self._lls.var(X)

    def gap_upper(self, indices):
        if self._ucb_estimates:
            return self._gap_upper_1(indices)
        else:
            return self._gap_upper_2(indices)

    def _gap_upper_1(self, indices):
        """
        Regret upper bound, computed as the ucb of the worst difference vector
        """
        X = self._game.get_actions(indices)
        D = difference_matrix(X)

        # compute ucb score for all differences and max out columns
        gaps = np.max(self._lls.ucb(D, delta=self._delta), axis=0)

        if self._truncate:
            gaps = np.minimum(gaps, 1)

        return gaps

    def _gap_upper_2(self, indices):
        """
        Regret upper bound, computed as the max_ucb - mean(x). This is a factor 2 conservative approximation.
        Does NOT work for partial monitoring games without modification.
        """
        X = self._game.get_actions(indices)
        max_ucb = np.max(self._lls.ucb(X, delta=self._delta))
        gaps = max_ucb - self._lls.mean(X)

        if self._truncate:
            gaps = np.minimum(gaps, 1)

        return gaps

    def gap_lower_2(self, indices):
        """
        Relaxed regret lower bound
        """
        X = self._game.get_actions(indices)
        D = difference_matrix(X)

        # compute ucb score for all differences and max out rows
        regret = np.max(self._lls.lcb(D, delta=self._delta), axis=0)

        return regret
