import numpy as np
from scipy.linalg import cho_solve, cho_factor

from pm.strategies.lls import RegularizedLeastSquares
from pm.strategy import Strategy
from pm.utils import difference_matrix


class DoubleRobustLLS(RegularizedLeastSquares):

    def add_data(self, m, x, y):
        super().add_data(x - m, y.flatten())

    def beta(self, delta=None):
        _s = max(self.s, 2)
        if delta is None:
            delta = 1 / (_s * np.log(_s))

        return (1 + np.sqrt(self._d * np.log(1 + self.s/self._d) + 2*np.log(self.s/delta)))**2


class Bose(Strategy):
    def __init__(self, game, robust_lls, delta):
        self._game = game
        self._robust_lls = robust_lls
        self.delta = delta
        self._m = None

    def add_observations(self, indices, y):
        x = self._game.get_actions(indices)
        self._robust_lls.add_data(self._m, x, y)

    def get_action(self):
        # compute plausible actions
        d = self._game.d
        indices = self._game.get_indices()
        k = self._game.k
        X = self._game.get_actions(indices)
        diff_X = difference_matrix(X).reshape(-1, d)
        diff_lcb = self._robust_lls.lcb(diff_X, delta=self.delta).reshape(k,k)
        diff_lcb_max = np.max(diff_lcb, axis=0)
        plausible_mask = (diff_lcb_max <= 1e-10)

        X_plausible = X[plausible_mask]
        I_plausible = indices[plausible_mask]
        if len(I_plausible) == 1:
            return I_plausible[0]

        var_Xp = self._robust_lls.var(X_plausible)
        chol_sol = cho_solve(self._robust_lls.get_cholesky_factor(), X_plausible.T)
        T = 400
        w = np.ones(len(I_plausible))/len(I_plausible)
        w_avg = np.zeros(len(I_plausible))
        count = 0

        # exponentiaed gradient descent
        for t in range(T):
            x_mean = w.dot(X_plausible)
            X_centered = X_plausible - x_mean
            var_x_xmean = self._robust_lls.var(X_centered)
            # best response
            best_i = np.argmax(var_x_xmean)
            grad = 2. * (x_mean - X_plausible[best_i]).dot(chol_sol) - var_Xp
            # loss gradient
            w *= np.exp(-np.sqrt(np.log(d)/T)*grad)
            w /= np.sum(w)
            if t > T/2:
                w_avg += w
                count += 1

        w_avg /= count
        x_mean = w_avg.dot(X_plausible)

        self._m = x_mean
        return np.random.choice(I_plausible, p=w_avg)
