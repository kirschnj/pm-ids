import unittest
from numpy.testing import assert_almost_equal
import numpy as np
from pm.estimator import RegularizedLeastSquares


class TestLLS(unittest.TestCase):


    def test_empty_lls(self):
        lls = RegularizedLeastSquares(d=2)

        self.assertEqual(lls._theta.shape, (2,))

        assert_almost_equal(lls.mean(np.array([1, 1])), np.zeros(1))
        assert_almost_equal(lls.mean(np.array([1,1,2,2,3,3]).reshape(3,2)), np.zeros(3))
        assert_almost_equal(lls.mean(np.array(np.arange(12)).reshape(2,3,2)), np.zeros(6).reshape(2,3))
        assert_almost_equal(lls.theta(), np.zeros(2))

    def test_lls(self):
        lls = RegularizedLeastSquares(d=2)
        lls.add_data(np.array([[1,1],[1,-1], [1,0]]) , np.array([1,2, 0]))
        assert_almost_equal(lls.theta(), np.array([0.75, -1/3]))
        assert_almost_equal(lls.mean(np.array([[1,1],[1,-1], [1,0]])), np.array([0.4166666667, 1.08333333333, 0.75000]))

    def test_beta(self):
        lls = RegularizedLeastSquares(d=2)
        lls.add_data(np.array([[1,1],[1,-1], [1,0]]) , np.array([1,2, 0]))

        delta = 0.5
        V = np.diag([4,3])
        logdet = np.log(np.linalg.det(V))
        beta = np.sqrt(logdet - 2*np.log(delta)) + 1
        assert_almost_equal(lls.beta(delta), beta)

    def test_var(self):
        lls = RegularizedLeastSquares(d=2)
        lls.add_data(np.array([[1, 1], [1, -1], [1, 0]]), np.array([1, 2, 0]))

        assert_almost_equal(lls.var(np.array([[1,0],[0,1], [1,1]])), np.array([0.25, 1/3, 0.25+1/3]))
if __name__ == '__main__':
    unittest.main()
