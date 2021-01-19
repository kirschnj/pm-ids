from pm.strategy import Strategy
import numpy as np
from scipy.linalg import cho_solve, cho_factor

from pm.utils import difference_matrix, psd_norm_squared
import cvxpy as cp


def full(indices, game, estimator):
    """
    log det(I + A^T V_t^{-1} A)
    """

    # first concatenate all A_ = [A_1, ..., A_l]
    # then compute solution of cholesky factor L^{-1} A
    m = game.m
    d = game.d

    A = game.get_observation_maps(indices)
    B = cho_solve(estimator.lls.get_cholesky_factor(), A.reshape(-1, d).T).T.reshape(-1, m, d)
    # multiply cholesky factors and add unit matrix to get: A^\T V^{-1} A + eye(m)
    C = np.matmul(A, np.swapaxes(B, -2, -1)) + np.eye(m)

    # iterate over matrices and compute determinant
    infogain = np.zeros(len(indices))
    for i, Ci in enumerate(C):
        L = cho_factor(Ci)[0]
        infogain[i] = 2*np.sum(np.log(np.diag(L)))


    lower_bound = estimator.gap_lower_2(indices)
    return infogain

def directeducb(indices, game, estimator):
    """
    First the most uncertain direction w in the set of plausible maximizers is computed,
    then I = log var(w) - log var(w|A)

    """
    m = game.m
    d = game.d
    I = game.get_indices()
    A = game.get_observation_maps(indices)

    # tentative update to V
    VA = np.matmul(np.swapaxes(A, -2, -1), A) + estimator.lls.V

    # compute plausible maximizer set
    ucb = estimator.ucb(I)
    w = game.get_actions(I[np.argmax(ucb)])

    # print(estimator.ucb([i]) - np.max(estimator.lcb(indices)))
    # print(estimator.ucb(indices) - np.max(estimator.lcb(indices)), 'ucb-max lcb')
    # print(estimator.regret_lower_2([i, j]))
    # print(w, i,j)
    # print(i,j)

    # if w is 0-vector, return 0 info gain

    log_var_w = np.log(estimator.lls.var(w))
    log_var_wA = np.zeros(len(indices))

    # compute log(var(w|A_i)) for each A_i
    for i, VAi in enumerate(VA):
        L = cho_factor(VAi)
        Vinv_w = cho_solve(L, w)
        log_var_wA[i] = np.log(w.dot(Vinv_w))

    return log_var_w - log_var_wA

def directed2(indices, game, estimator):
    """
    First the most uncertain direction w in the set of plausible maximizers is computed,
    then I = log var(w) - log var(w|A)

    """
    m = game.m
    d = game.d
    I = game.get_indices()
    A = game.get_observation_maps(indices)

    # tentative update to V
    VA = np.matmul(np.swapaxes(A, -2, -1), A) + estimator.lls.V

    # compute plausible maximizer set
    lower_bound = estimator.gap_lower_2(I)
    # print(lower_bound)
    P = game.get_actions(I[lower_bound <= 10e-10])
    # all pairs of plausible maximizers
    PP = difference_matrix(P)
    i, j = np.unravel_index(np.argmax(estimator.lls.var(PP)), (len(P), len(P)))

    w = P[i] - P[j]

    # print(estimator.ucb([i]) - np.max(estimator.lcb(indices)))
    # print(estimator.ucb(indices) - np.max(estimator.lcb(indices)), 'ucb-max lcb')
    # print(estimator.regret_lower_2([i, j]))
    # print(w, i,j)
    # print(i,j)

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

def plausible_maximizers_2(game, estimator):
    I = game.get_indices()
    lower_bound = estimator.gap_lower_2(I)
    return I[lower_bound <= 10e-10]


def directed3(indices, game, estimator):
    """
    First the most uncertain direction w in the set of plausible maximizers is computed,
    then I = log var(w) - log var(w|A)

    """
    m = game.m
    d = game.d
    I = game.get_indices()
    A = game.get_observation_maps(indices)

    # tentative update to V
    VA = np.matmul(np.swapaxes(A, -2, -1), A) + estimator.lls.V

    # compute plausible maximizer set


    P = game.get_actions(I[np.max(estimator.lcb(I)) <= estimator.ucb(I)])

    max_var = -1
    max_w = None
    for x in P:
        for y in P:
            tvar = estimator.lls.var(x-y)
            if tvar > max_var:
                max_var = tvar
                max_w = x-y

    w = max_w

    # all pairs of plausible maximizers
    # PP = difference_matrix(P)
    # i, j = np.unravel_index(np.argmax(estimator.lls.var(PP)), (len(P), len(P)))
    #
    # w = P[i] - P[j]
    #
    # # print(estimator.ucb([i]) - np.max(estimator.lcb(indices)))
    # # print(estimator.ucb(indices) - np.max(estimator.lcb(indices)), 'ucb-max lcb')
    # # print(estimator.regret_lower_2([i, j]))
    # # print(w, i,j)
    # print(i,j)

    # if w is 0-vector, return 0 info gain
    if np.sum(w*w) <= 10e-20:
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
        self._update_estimator = True
        self._t = 1  # step counter

    def get_next_action(self):
        indices = self._game.get_indices()
        gaps = self._estimator.gap_upper(indices)
        infogain = self._infogain(indices, self._game, self._estimator)

        if self._deterministic:
            ratio = gaps ** 2 / infogain
            return indices[np.argmin(ratio)]
        else:
            return self._ids_sample(indices, gaps, infogain)

    def add_observations(self, indices, y):
        if self._update_estimator:
            super().add_observations(indices, y)

        self._t += 1  # increase step counter each time we get the data

    def _mixed_ratio(self, A, B, C, D, p_new, ratio, p):

        # invalid x, return previous ratio
        if p_new < 0 or p_new > 1:
            return ratio, p

        # if the ratio is better with x, return new ratio and x
        tmp_ratio = (p_new * A + (1 - p_new) * B) ** 2 / (p_new * C + (1 - p_new) * D)
        if tmp_ratio < ratio:
            return tmp_ratio, p_new

        return ratio, p

    def _ids_sample(self, indices, gaps, infogain):
        """
        Compute the randomized IDS solution
        https://www.wolframalpha.com/input/?i=d%2Fdx+(Ax+%2B+(1-x)*B)^2%2F(Cx+%2B+(1-x)D)+
        """
        best_p = None
        best_i = None
        best_j = None
        best_ratio = 10e10

        for i in range(len(indices)):
            for j in range(len(indices)):
                ratio = 10e10
                p = None
                A, B, C, D = gaps[i], gaps[j], infogain[i], infogain[j]

                # deterministic on i
                if infogain[i] > 0:
                    ratio, p = self._mixed_ratio(A, B, C, D, 1., ratio, p)

                # deterministic on j
                if infogain[j] > 0:
                    ratio, p = self._mixed_ratio(A, B, C, D, 0., ratio, p)

                # mixed solution
                if np.abs(C - D) > 10e-20 and np.abs(B - A) > 10e-20:
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

    def id(self):
        """
        returns identifier.
        ID is {ids,dids}-infogain
        """
        _id = "ids"
        if self._deterministic:
            _id = "dids"
        _id += f"-{self._infogain.__name__}"

        return _id


class AsymptoticIDS(IDS):

    def __init__(self, game, estimator):
        super().__init__(game, estimator, infogain=None, deterministic=False)

    def _info_game(self, indices, winner, nu, beta, V_norm):
        # compute mixing weights
        eta = 1. / np.sqrt(beta)
        q = np.exp(-eta * V_norm)
        q[winner] = 0.

        # compute info gain
        X = self._game.get_actions(indices)
        I = q @ ((nu - self._estimator.lls.theta) @ X.T)**2  # compute sum_y q(y) <nu(y) - theta, x>^2 for each x
        return I

    def compute_nu(self, indices):
        """
        Compute the alternative parameters and Vnorm for each cell
        """
        d = self._game.d
        X = self._game.get_actions(indices)
        theta = self._estimator.lls.theta
        V = self._estimator.lls.V
        nu = np.zeros((len(indices), d))
        C = difference_matrix(X)
        # for each action, solve the quadratic program to find the alternative
        for i in indices:
            x = cp.Variable(d)
            q = -2 * (V @ theta)
            G = -C[i, :, :]

            prob = cp.Problem(cp.Minimize(cp.quad_form(x, V) + q.T @ x), [G @ x <= 0])
            prob.solve()

            nu[i:] = x.value

        # check corner cases in the bandit case : can the projected nu have a very large norm ? => regularization ?
        # normalize as per our unit ball hypothesis => creates bugs when the projection on the cone is too close to origin. Also does it make sense ?
        # nu /= np.linalg.norm(nu, axis=1)[:, None]
        V_norm = psd_norm_squared(nu - theta, V)
        return nu, V_norm

    def get_next_action(self):
        """
        Compute the IDS solution when there's "enough" data.
        Deprecated : and otherwise fallback on UCB
        """
        indices = self._game.get_indices()
        #we may want to try other values for beta_t
        beta_t = self._estimator.lls.beta(1/(self._t * np.log(self._t + 2)))  # adding +1 to avoid numerical issues at initialization

        # only re-compute quantities when the estimator changes
        if self._update_estimator:
            # compute winner
            means = self._estimator.lls.mean(self._game.get_actions(indices))
            self._winner = np.argmax(means)

            # compute alternatives and V_norm
            self._nu, self._V_norm = self.compute_nu(indices)
            V_norm_tmp = self._V_norm.copy()

            # compute minimum V-norm without winner
            V_norm_tmp[self._winner] = np.Infinity
            self._min_V_norm = np.min(V_norm_tmp)


        winner, V_norm, nu = self._winner, self._V_norm, self._nu

        # check exploration/exploitation condition
        if self._min_V_norm < beta_t:
            self._update_estimator = True  # exploration => collect data

            gaps = self._estimator.gap_upper(indices)
            delta_s = gaps[winner]  # smallest gap

            # print()
            # print(gaps)
            # print(delta_s)
            # check IDS condition 2 delta_s <= ^Delta(y)
            # if np.sum(gaps < 2 * delta_s) == 1:
                # print(f"explore ids {self._t}")

            infogain = self._info_game(indices, winner, nu, beta_t, V_norm)
            return self._ids_sample(indices, gaps, infogain)
            # else:
            #     # print(f"explore ucb {self._t}")
            #     # play UCB
            #     ucbs = self._estimator.ucb(indices)
            #     return indices[np.argmax(ucbs)]
        else:
            # print("exploit")
            self._update_estimator = False
            return winner

    def id(self):
        return "asymptotic_ids"
