import unittest
import numpy as np
from numpy.testing import assert_almost_equal

from pm.estimator import RegularizedLeastSquares, RegretEstimator
from pm.games.bandit import Bandit
from pm.games.pm import GenericPM
from pm.strategies.ids import full, directed2
from scipy.linalg import cho_solve

class MyTestCase(unittest.TestCase):

    def test_full_infogain(self):
        X = np.arange(10).reshape(5,2)
        Y = np.arange(5)

        lls = RegularizedLeastSquares(2)
        lls.add_data(X, Y)

        V = X.T.dot(X) + np.eye(2)

        game = Bandit(X=X)
        I = game.get_indices()
        estimator = RegretEstimator(game, lls, delta=0.5, truncate=False)

        assert_almost_equal(full(I[0:1], game, estimator), np.log(1 + lls.var(X[0:1])))
        assert_almost_equal(full(I[0:4], game, estimator), np.log(1 + lls.var(X[0:4])))

        A = np.arange(30).reshape(5,3,2)
        game = GenericPM(X, A)

        i0 = np.log(np.linalg.det(np.eye(3) + A[0].dot(np.linalg.solve(V, A[0].T))))
        i1 = np.log(np.linalg.det(np.eye(3) + A[1].dot(np.linalg.solve(V, A[1].T))))
        assert_almost_equal(full([0], game, estimator), i0)
        assert_almost_equal(full([0, 1], game, estimator), [i0, i1])


    def test_directed_infogain(self):
        X = np.arange(10).reshape(5, 2)
        Y = np.arange(5)
        A = np.arange(30).reshape(5, 3, 2)
        game = GenericPM(X, A)

        lls = RegularizedLeastSquares(2)
        lls.add_data(0.01*X, Y)
        V = 0.01*X.T.dot(0.01*X) + np.eye(2)

        estimator = RegretEstimator(game, lls, delta=0.5)

        # compute most uncertain plausible maximizer
        max_var = -1.
        max_w = None

        for i in range(5):
            for j in range(5):
                w = X[i] - X[j]
                # check if i and j are plausible maximizers
                if np.all(estimator.gap_lower_2([i, j]) <= 10e-10):
                    if lls.var(w) > max_var:
                        max_var = lls.var(w)
                        max_w = w

        # compute the reference info gain
        i0 = np.log(lls.var(max_w)) - np.log(max_w.dot(np.linalg.solve(V + A[0].T.dot(A[0]), max_w.T)))
        i1 = np.log(lls.var(max_w)) - np.log(max_w.dot(np.linalg.solve(V + A[1].T.dot(A[1]), max_w.T)))


        assert_almost_equal(directed2([0], game, estimator), [i0])
        assert_almost_equal(directed2([0, 1], game, estimator), [i0, i1])

    def test_info_game(self):
        X = np.arange(10).reshape(5,2)
        Y = np.arange(5)

        lls = RegularizedLeastSquares(2)
        lls.add_data(X, Y)

        V = X.T.dot(X) + np.eye(2)

        game = Bandit(X=X)
        I = game.get_indices()
        estimator = RegretEstimator(game, lls, delta=0.5, truncate=False)

        # assert_almost_equal(full(I[0:1], game, estimator), np.log(1 + lls.var(X[0:1])))
        # assert_almost_equal(full(I[0:4], game, estimator), np.log(1 + lls.var(X[0:4])))
        #




if __name__ == '__main__':
    unittest.main()
