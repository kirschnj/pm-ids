import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular


from pm.utils import difference_matrix


class RegularizedLeastSquares:

    def __init__(self, d, reg=1, beta_logdet=True, noise_var=1., scale_obs=False, beta_factor=1., beta=None, delta=None):
        self._d = d
        self._V = np.eye(self._d)
        self._XY = np.zeros(self._d)
        self._theta = np.zeros(self._d)
        self._s = 1
        self.delta = delta

        self._user_beta = beta
        self.beta_logdet = beta_logdet
        self.beta_factor = beta_factor
        self.noise_var = noise_var
        self.noise_std = np.sqrt(self.noise_var)

        self.scale_obs = scale_obs  # if true, feature vectors and observation are scaled by 1/noise_std (used to compute the posterior for TS)
        self._update_cache()

    def add_data(self, x, y):
        if self.scale_obs:
            x = x/self.noise_std
            y = y/self.noise_std
        self._V += x.T.dot(x)
        self._XY += x.T.dot(y)
        self._update_cache()
        self._s += len(y)

    @property
    def theta(self):
        """ estimated parameter """
        return self._theta

    @property
    def cholesky_factor(self):
        return self._cholesky

    @property
    def V(self):
        """ co-variance matrix of the estimator """
        return self._V

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

    def cw(self, x, delta=None):
        """
        Confidence width
        """
        return np.sqrt(self.beta(delta) * self.var(x))

    def ucb(self, x, delta=None):
        """
        Upper confidence bound for x at confidence level delta
        """
        return self.mean(x) + self.cw(x, delta)

    def lcb(self, x, delta=None):
        """
        Upper confidence bound for x at confidence level delta
        """
        return self.mean(x) - self.cw(x, delta)

    def lcb_ucb(self, x, delta=None):
        cw = self.cw(x, delta)
        mean = self.mean(x)
        return mean - cw, mean + cw

    def beta(self, delta=None):
        """
        compute beta = sqrt(log det(V_t) + 2 log(1/delta)) + 1
        :param delta:
        :return:
        """
        if self._user_beta is not None:
            return self._user_beta

        _s = max(self.s, 2)

        if delta is None:  # delta not passed to the function
            delta = self.delta
            if delta is None:  # global default is None
                delta = 1/(_s*np.log(_s))

        noise_std_factor = self.noise_std if not self.scale_obs else 1.

        if self.beta_logdet:
            logdet = 2 * np.sum(np.log(np.diag(self._cholesky[0])))
            beta = (noise_std_factor*np.sqrt(logdet + 2*np.log(1/delta)) + 1)**2
        else:
            beta = noise_std_factor**2 * 2* np.log(1/delta) + self._d * max(np.log(np.log(_s)), 1)
        return self.beta_factor * beta

    def _update_cache(self):
        self._cholesky = cho_factor(self._V)
        self._theta = cho_solve(self._cholesky, self._XY)


    def posterior_samples(self, size=1):
        eta = np.random.normal(0, 1, self._d*size).reshape(size, self._d)
        sample = self.theta + solve_triangular(self.cholesky_factor[0], eta.T).T
        return sample

#
# class RegretEstimator:
#
#     def __init__(self, game, lls, truncate=True, delta=None, ucb_estimates=True):
#         self._truncate = truncate
#         self._game = game
#         self._d = self._game.d
#         self._lls = lls
#         self._delta = delta
#         self._ucb_estimates = ucb_estimates  # this flag determines the way the gap upper bound is computed
#
#     @property
#     def lls(self):
#         return self._lls
#
#     def add_data(self, indices, y):
#         # stack the observations and observation operators
#         ax = self._game.get_observation_maps(indices)
#         ax = np.vstack(ax)
#         y = np.hstack(y)
#
#         # update lls
#         self._lls.add_data(ax, y)
#
#     def ucb(self, indices, delta=None):
#         X = self._game.get_actions(indices)
#
#         # if self._truncate:
#         #     return np.minimum(self._lls.ucb(X, delta=self._delta), 1)
#         if delta == None:
#             return self._lls.ucb(X, delta=self._delta)
#         else:
#             return self._lls.ucb(X, delta=delta)
#
#     def lcb(self, indices,):
#         X = self._game.get_actions(indices)
#         return self._lls.lcb(X, delta=self._delta)
#
#     def var(self, indices):
#         X = self._game.get_actions(indices)
#         return self._lls.var(X)
#
#     def gap_upper(self, indices):
#         if self._ucb_estimates:
#             return self._gap_upper_1(indices)
#         else:
#             return self._gap_upper_2(indices)
#
#     def _gap_upper_1(self, indices):
#         """
#         Regret upper bound, computed as the ucb of the worst difference vector
#         """
#         X = self._game.get_actions(indices)
#         D = difference_matrix(X)
#
#         # compute ucb score for all differences and max out columns
#         gaps = np.max(self._lls.ucb(D, delta=self._delta), axis=0)
#
#         if self._truncate:
#             gaps = np.minimum(gaps, 1)
#
#         return gaps
#
#     def _gap_upper_2(self, indices):
#         """
#         Regret upper bound, computed as the max_ucb - mean(x). This is a factor 2 conservative approximation.
#         Does NOT work for partial monitoring games without modification.
#         """
#         X = self._game.get_actions(indices)
#         max_ucb = np.max(self._lls.ucb(X, delta=self._delta))
#         gaps = max_ucb - self._lls.mean(X)
#
#         if self._truncate:
#             gaps = np.minimum(gaps, 1)
#
#         return gaps
#
#     def gap_lower_2(self, indices):
#         """
#         Relaxed regret lower bound
#         """
#         X = self._game.get_actions(indices)
#         D = difference_matrix(X)
#
#         # compute ucb score for all differences and max out rows
#         regret = np.max(self._lls.lcb(D, delta=self._delta), axis=0)
#
#         return regret
