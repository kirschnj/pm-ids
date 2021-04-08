from pm.game import Game
from pm.strategies.lls import RegularizedLeastSquares
from pm.strategy import Strategy
import numpy as np

from pm.utils import difference_matrix


class GCB(Strategy):
    def __init__(self, game : Game, lls : RegularizedLeastSquares):#, gap_estimator :GapEstimator):
        """
        """
        super().__init__(game)
        self.lls = lls
        # self.gap_estimator = gap_estimator
        self.obs_set, self.explore_num, self.V = game.get_global_obs_set()
        self.t = 0
        self.theta = np.zeros(game.d)
        self._theta = np.empty((0,game.d))
        self._MY = np.zeros(game.d)
        self.nsigma = 0
        self.alpha = 1
        self.explore_i = 0
        self.exploit = False


    def ft(self):
        _t = max(self.t, 2)
        return np.log(_t) + 2 * np.log(self.game.k)

    def get_action(self):
        # self.gap_estimator.estimate(self.game, self.lls)

        X = self.game.X
        # finished one round of exploration
        if self.explore_i == self.explore_num:
            means = X @ self.theta
            self.winner = winner = np.argmax(means)
            means_copy = means.copy()
            means_copy[winner] = -np.inf
            second = np.argmax(means_copy)
            # print(winner, second, means)
            # print(means[winner] - means[second], np.sqrt(self.ft() * self.alpha / self.nsigma), self.nsigma, self.t ** (2 / 3) * self.ft())
            if means[winner] - means[second] > np.sqrt(self.ft()*self.alpha/self.nsigma) \
                    or self.nsigma > self.t**(2/3)*self.ft():
                self.exploit = True
            else:
                self.exploit = False
                self.explore_i = 0

        if self.exploit:
            # print("exploit")
            return self.winner
        else:
            # print("explore")
            return self.obs_set[self.explore_i]

    def add_observations(self, actions, observations):
        self.t += 1
        if self.exploit:
            return

        m = self.game.get_observation_maps()[actions]
        # print(observations @ m)
        self._MY += observations @ m
        self.explore_i += 1

        # print(self._MY)
        if self.explore_i == self.explore_num:
            # theta = np.linalg.solve(self.V, self._MY).reshape(1,-1)
            theta = np.linalg.lstsq(self.V, self._MY, rcond=None)[0].reshape(1, -1)
            self._theta = np.vstack((self._theta, theta))
            self.theta = np.mean(self._theta, axis=0)
            self.nsigma += 1
            # print(self.theta)
            self._MY = np.zeros(self.game.d)
