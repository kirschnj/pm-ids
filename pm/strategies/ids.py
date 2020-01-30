from pm.strategy import Strategy
import numpy as np
from scipy.linalg import cho_solve, cho_factor

from pm.utils import difference_matrix


def full_info(indices, game, estimator):
    """
    log det(I + A^T V_t^{-1} A)
    """

    # first concatenate all A_ = [A_1, ..., A_l]
    # then compute solution of cholesky factor L^{-1} A
    m = game.get_m()
    d = game.get_d()

    A = game.get_observation_maps(indices)
    B = cho_solve(estimator.lls.get_cholesky_factor(), A.reshape(-1, d).T).T.reshape(-1, m, d)
    # multiply cholesky factors and add unit matrix to get: A^\T V^{-1} A + eye(m)
    C = np.matmul(A, np.swapaxes(B, -2, -1)) + np.eye(m)

    # iterate over matrices and compute determinant
    infogain = np.zeros(len(indices))
    for i, Ci in enumerate(C):
        L = cho_factor(Ci)[0]
        infogain[i] = 2*np.sum(np.log(np.diag(L)))

    return infogain


def directed_info_2(indices, game, estimator):
    """
    First the most uncertain direction w in the set of plausible maximizers is computed,
    then I = log var(w) - log var(w|A)

    """
    m = game.get_m()
    d = game.get_d()
    I = game.get_indices()
    A = game.get_observation_maps(indices)

    # tentative update to V
    VA = np.matmul(np.swapaxes(A, -2, -1), A) + estimator.lls.get_V()

    # compute plausible maximizer set
    lower_bound = estimator.regret_lower_2(I)
    P = game.get_actions(I[lower_bound <= 10e-10])
    # all pairs of plausible maximizers
    PP = difference_matrix(P)
    i, j = np.unravel_index(np.argmax(estimator.lls.var(PP)), (len(P), len(P)))
    w = P[i] - P[j]

    # if w is 0-vector, return 0 info gain
    if np.sum(w*w) <= 10e-30:
        return np.zeros(len(indices))

    log_var_w = np.log(estimator.lls.var(w))
    log_var_wA = np.zeros(len(indices))

    # compute log(var(w|A_i)) for each A_i
    for i, VAi in enumerate(VA):
        L = cho_factor(VAi)
        Vinv_w = cho_solve(L, w)
        log_var_wA[i] = np.log(w.dot(Vinv_w))

    return log_var_w - log_var_wA


class IDS(Strategy):

    def __init__(self, game, estimator, infogain, deterministic=False):
        super().__init__(game, estimator)
        self._infogain = infogain
        self._deterministic = deterministic

    def get_next_action(self):
        if self._deterministic:
            return self._deterministic_ids()
        else:
            return self._ids()

    def _mixed_ratio(self, A, B, C, D, p_new, ratio, p):

        # invalid x, return previous ratio
        if p_new < 0 or p_new > 1:
            return ratio, p

        # if the ratio is better with x, return new ratio and x
        tmp_ratio = (p_new * A + (1 - p_new) * B) ** 2 / (p_new * C + (1 - p_new) * D)
        if tmp_ratio < ratio:
            return tmp_ratio, p_new

        return ratio, p

    def _ids(self):
        """
        Compute the randomized IDS solution
        https://www.wolframalpha.com/input/?i=d%2Fdx+(Ax+%2B+(1-x)*B)^2%2F(Cx+%2B+(1-x)D)+
        """
        indices = self._game.get_indices()
        regret = self._estimator.regret(indices)
        infogain = self._infogain(indices, self._game, self._estimator)

        best_p = None
        best_i = None
        best_j = None
        best_ratio = 10e10

        for i in range(len(indices)):
            for j in range(len(indices)):
                ratio = 10e10
                p = None
                A, B, C, D = regret[i], regret[j], infogain[i], infogain[j]

                # deterministic on i
                if infogain[i] > 0:
                    ratio, p = self._mixed_ratio(A, B, C, D, 1., ratio, p)

                # deterministic on j
                if infogain[j] > 0:
                    ratio, p = self._mixed_ratio(A, B, C, D, 0., ratio, p)

                # mixed solution
                if np.abs(C - D) > 10e-20 and np.abs(C - D) > 10e-20:
                    x = D/(C-D)
                    ratio, p = self._mixed_ratio(A, B, C, D, x, ratio, p)

                    x = B/(B-A)
                    ratio, p = self._mixed_ratio(A, B, C, D, x, ratio, p)

                    x = (2*A*D - B*C - B*D)/(B-A)/(C-D)
                    ratio, p = self._mixed_ratio(A, B, C, D, x, ratio, p)

                if ratio < best_ratio:
                    best_ratio = ratio
                    best_i, best_j = i, j
                    best_p = p

                if i == j:
                    break

        if np.random.binomial(1, p=best_p):
            return indices[best_i]
        else:
            return indices[best_j]

    def _deterministic_ids(self):
        """
        Compute the deterministic IDS solution
        """
        indices = self._game.get_indices()
        ratio = np.square(self._estimator.regret(indices))/self._infogain(indices, self._game, self._estimator)
        return indices[np.argmin(ratio)]

    def id(self):
        if self._deterministic:
            return "dids"
        return "ids"
