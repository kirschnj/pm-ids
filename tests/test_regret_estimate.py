import unittest
import numpy as np
from numpy.testing import assert_almost_equal

from pm.estimator import RegularizedLeastSquares, RegretEstimator
from pm.games.bandit import Bandit

class MyTestCase(unittest.TestCase):
    def test_regret_estimate(self):
        X = np.arange(10).reshape(5,2)
        Y = np.arange(5)

        lls = RegularizedLeastSquares(2)
        lls.add_data(X, Y)

        mean = lls.mean(X)
        assert_almost_equal(mean, np.array([0.14403292, 1.09053498, 2.03703704, 2.98353909, 3.93004115]))

        var = lls.var(X)
        assert_almost_equal(var, np.array([0.24897119, 0.15020576, 0.16666667, 0.29835391, 0.54526749]))

        ucb = lls.ucb(X, delta=0.5)
        assert_almost_equal(ucb, np.array([2.01607811, 2.54460417, 3.56871037, 5.03284902, 6.70046731]))

        game = Bandit(X=X)
        I = game.get_indices()
        regret_estimate = RegretEstimator(game, lls, delta=0.5, truncate=False)

        assert_almost_equal(regret_estimate.regret_upper(I), np.array([0.        , 1.84704133, 3.69408266, 5.541124  , 7.38816533]))


if __name__ == '__main__':
    unittest.main()
