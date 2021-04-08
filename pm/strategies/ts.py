from scipy.linalg import solve_triangular

from pm.strategies.lls import RegularizedLeastSquares
from pm.strategy import Strategy
import numpy as np

class TS(Strategy):
    def __init__(self, game, lls : RegularizedLeastSquares):
        super().__init__(game)
        self.lls = lls

    def get_action(self):
        sample = self.lls.posterior_samples(size=1).reshape(self.game.d)
        X = self.game.X
        reward = X.dot(sample)
        return np.argmax(reward)

    def add_observations(self, actions, obs):
        m = self.game.M[actions]
        # if self._force_homoscedastic:
        #     x = self.game.X[actions]
        #     rho = np.linalg.norm(x)/np.linalg.norm(m)
        #     m = x.reshape(len(obs),-1)
        #     obs = obs * rho
        self.lls.add_data(m, obs)