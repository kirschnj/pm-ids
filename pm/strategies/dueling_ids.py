import logging
import numpy as np
import cvxpy as cp
from pm.strategies.gp import DuelingGPRegression
from scipy.linalg import cho_solve, cho_factor
import osqp
from scipy import sparse
from GPy.kern import RBF

from pm.strategies.ids import IDS
from pm.utils import difference_matrix, psd_norm_squared
from pm.strategy import Strategy

class DuelingIDS(IDS):

    def __init__(self, game, estimator):
        super().__init__(game, estimator, infogain=None)

    def get_action(self):
        """
        Compute the IDS solution when there's "enough" data.
        """
        base_indices = self._game.get_base_indices()
        actions = self._game.get_base_actions(base_indices)

        lls = self._estimator.lls
        means = lls.mean(actions)
        winner = np.argmax(means)

        action_diff = actions - actions[winner]
        ucb_diff = lls.ucb(action_diff)
        delta = np.max(ucb_diff + means - means[winner])
        if delta <= 1e-10:
            return (base_indices[winner], base_indices[winner])

        gaps = delta + means[winner] - means

        infogain = np.log(1 + lls.var(action_diff))
        p_best = None
        x_best = None
        ratio_best = np.inf
        for x, D, I in zip(base_indices, gaps, infogain):
            p, ratio = self._two_action_ratio(delta, D, 0., I)
            # print(p, ratio)
            if ratio < ratio_best:
                ratio_best = ratio
                p_best = p
                x_best = x

        if np.random.binomial(1, p_best):
            return (base_indices[winner], x_best)
        else:
            return (base_indices[winner], base_indices[winner])


class DuelingKernelIDS(IDS):

    def __init__(self, game, delta, reg=1., beta=None, lengthscale=1.):
        super().__init__(game, None, infogain=None)
        self._x1train = np.empty((0, game.d))
        self._x2train = np.empty((0, game.d))
        self._dtrain = np.empty(0)
        self.t = 0
        self._rbf = RBF(game.d, lengthscale=lengthscale)
        self.delta = delta
        self.reg = reg
        self._beta = beta
        self._update = True

        if game.discrete_x is None:
            raise RuntimeError()
        self.X = game.discrete_x

    def kernel(self, x, y=None):
        return self._rbf.K(x, y)

    def add_observations(self, x, y):
        if not self._update:
            return
        assert len(x) == 1
        x = x[0]

        x1 = x[0]
        x2 = x[1]
        self._x1train = np.vstack([self._x1train, x1])
        self._x2train = np.vstack([self._x2train, x2])
        self._dtrain = np.concatenate([self._dtrain, y.flatten()])
        self.t += 1

    def get_action(self):
        if self.t == 0:
            i = np.random.choice(len(self.X) - 1)
            return self.X[i], self.X[i + 1]

        gp = DuelingGPRegression(self._x1train, self._x2train, self._dtrain, kernel=self.kernel, reg=self.reg)

        gp.precompute_target(self.X)
        mean = gp.mean()

        # print(mean)

        if self._beta is None:
            beta = gp.beta(self.delta)
        else:
            beta = self._beta


        winner = np.argmax(mean)
        x_best = self.X[winner]

        psi = np.maximum(gp.psi(x_best, self.X), 0)
        # print(psi)
        delta_t = np.max(mean - mean[winner] + np.sqrt(beta*psi))
        # print(delta_t)
        if delta_t <= 1e-10:
            self._update = False
            return (x_best, x_best)
        gaps = delta_t + mean[winner] - mean.flatten()
        infogain = np.log(1. + psi).flatten()

        p_best = None
        x_info = None
        ratio_best = np.inf
        # print(infogain)
        for D, I, x in zip(gaps, infogain, self.X):
            p, ratio = self._two_action_ratio(delta_t, D, 0., I)
            # print(p, ratio)
            if ratio < ratio_best:
                ratio_best = ratio
                p_best = p
                x_info = x

        if np.random.binomial(1, p_best):
            self._update = True
            return (x_best, x_info)
        else:
            self._update = False
            return (x_best, x_best)
