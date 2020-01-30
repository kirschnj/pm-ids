import unittest
import numpy as np
from numpy.testing import assert_almost_equal

from pm.estimator import RegularizedLeastSquares, RegretEstimator
from pm.games.bandit import Bandit
from pm.utils import difference_matrix


class MyTestCase(unittest.TestCase):

    def test_difference_matrix(self):
        X = np.arange(6).reshape(3,2)
        D = difference_matrix(X)


        self.assertEqual(D.shape, (3,3,2))
        for i in range(3):
            for j in range(3):
                assert_almost_equal(D[i,j], X[i] - X[j])

if __name__ == '__main__':
    unittest.main()
