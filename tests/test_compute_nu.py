import unittest
import numpy as np
from numpy.testing import assert_almost_equal

from pm.estimator import RegularizedLeastSquares, RegretEstimator
from pm.games.bandit import Bandit


# TODO: fix this test
# from pm.strategies.optids import compute_nu
from scipy.linalg import cho_solve

class MyTestCase(unittest.TestCase):

  def test_compute_nu(self):
      X = np.arange(10).reshape(5,2)
      Y = np.arange(5)

      lls = RegularizedLeastSquares(2)
      lls.add_data(X, Y)

      V = X.T.dot(X) + np.eye(2)

      game = Bandit(X=X)
      I = game.get_indices()
      estimator = RegretEstimator(game, lls, delta=0.5, truncate=False)

      # test that linear constraints are right
      assert_almost_equal(game.get_cell_constraints(0), np.array([[0,0],[1,-1],[-1,1],[0,2]]))

      #compute projections of theta_hat on the cell of each action
      nu = compute_nu(estimator, game)
      #todo : find a good bandit instance for this test
      # assert_almost_equal(nu, )
