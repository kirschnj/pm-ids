from pm.game import Game
from pm.strategies.lls import RegularizedLeastSquares
from pm.strategy import Strategy
import numpy as np

from pm.utils import difference_matrix


class MaxInP(Strategy):
    def __init__(self, game : Game, lls : RegularizedLeastSquares):#, gap_estimator :GapEstimator):
        """
        """
        super().__init__(game)
        self.lls = lls
        # self.gap_estimator = gap_estimator

    def get_action(self):
        # self.gap_estimator.estimate(self.game, self.lls)
        X = self.game.X_base
        base_k = len(X)
        diff_X = difference_matrix(X)
        ucb_diff_X = self.lls.ucb(diff_X.reshape(-1, self.game.d)).reshape(base_k, base_k)
        ucb_diff_X += np.diag(np.ones(base_k)*np.inf)
        # print(ucb_diff_X)
        plausible = np.min(ucb_diff_X, axis=1) > 0
        # print(diff_X.shape)
        plausible_diff = diff_X[plausible][:, plausible, :]
        # print(plausible)
        # print(plausible_diff)
        num_plausible = plausible_diff.shape[0]
        var = self.lls.var(plausible_diff.reshape(-1, self.game.d)).reshape(num_plausible, num_plausible)
        # print(var)
        most_uncertain = np.unravel_index(np.argmax(var), (num_plausible, num_plausible))

        indices = np.arange(base_k)[plausible]
        i1 = indices[most_uncertain[0]]
        i2 = indices[most_uncertain[1]]
        return base_k*i1 + i2
        # plausible_indices = indices[plausible][:,plausible]
        # print(plausible_indices[most_uncertain])
        # print(most_uncertain)

        # exit()
        # index_set = np.arange(self.game.k)
        # return (index_set[plausible_diff.flatten()])[most_uncertain]

    def add_observations(self, actions, observations):
        m = self.game.get_observation_maps()[actions]
        self.lls.add_data(m, observations)