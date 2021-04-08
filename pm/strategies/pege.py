from pm.game import Game
from pm.strategies.lls import RegularizedLeastSquares
from pm.strategy import Strategy
import numpy as np

class PEGE(Strategy):
    def __init__(self, game : Game, lls : RegularizedLeastSquares, mode='worst_case'):#, gap_estimator :GapEstimator):
        """
        """
        super().__init__(game)

        self.lls = lls
        if mode == 'worst_case':
            self.C = lambda x: np.log(x)
            self.alpha = .5
            self.beta = 0.
        elif mode == 'log':
            self.C = lambda x: x
            self.alpha = 1.
            self.beta = 1.
        else:
            raise ValueError("Invalid configuration")

        self.b = 1
        self.explore = True
        self.explore_i = 0
        self.j = 0

        # find a global observer set
        self.obs_set, self.explore_num, self.V = game.get_global_obs_set()

        self.theta = np.zeros(game.d)
        self._theta = np.empty((0,game.d))
        self._MY = np.zeros(game.d)

    def get_action(self):
        if self.explore:
            # print("explore")
            return self.obs_set[self.explore_i]
        else:
            # print("exploit")
            means = self.game.X @ self.theta
            return np.argmax(means)

    def add_observations(self, actions, observations):
        if self.explore:
            # store observation
            m = self.game.get_observation_maps()[actions]
            self._MY += observations @ m

            # go to next exploration action
            self.explore_i += 1

            # cycle through observer set
            if self.explore_i == self.explore_num:
                self.explore_i = 0
                self.j += 1  # next iteration

            # stop exploration phase
            if self.j > self.b**self.beta:
                self.explore = False

                theta = np.linalg.lstsq(self.V, self._MY, rcond=None)[0].reshape(1, -1) / self.j
                self._theta = np.vstack((self._theta, theta))
                self.theta = np.mean(self._theta, axis=0)
                # print(self.theta)
                self._MY = np.zeros(self.game.d)
                self.j = 0
        else:
            self.j += 1
            # stop exploitation phase
            if self.j > np.exp(self.C(self.b**self.alpha)):
                # print(self.j)
                # print(self.b**self.alpha)
                # print(self.C(self.b**self.alpha))
                self.j = 0
                self.explore = True
                self.b += 1
        # print(self.theta)
