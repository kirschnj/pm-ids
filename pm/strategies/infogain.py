from abc import ABC

from scipy.linalg import cho_solve, cho_factor, solve_triangular
import numpy as np
from scipy.stats import multivariate_normal
from pm.strategies.gaps import GapEstimator
from pm.utils import difference_matrix


class InfoGain(ABC):
    def info(self, gap_estimator : GapEstimator, context=None):
        raise NotImplemented

class WorstCaseInfoGain(InfoGain):

    def info(self, gap_estimator : GapEstimator, context=None):
        """
        log det(I + A^T V_t^{-1} A)
        """

        # first concatenate all A_ = [A_1, ..., A_l]
        # then compute solution of cholesky factor L^{-1} A
        game = gap_estimator.game
        lls = gap_estimator.lls
        m = game.m
        d = game.d
        M = game.M
        if context is not None:
            M = M[context]

        # the one-dimensional case can be computed directly
        if m == 1:
            return np.log(1 + lls.var(M.reshape(-1,d)))

        # if we have more than one output dimension, we need to compute the eigenvalues
        B = cho_solve(lls.cholesky_factor, M.reshape(-1, d).T).T.reshape(-1, m, d)
        # multiply cholesky factors and add unit matrix to get: A^\T V^{-1} A + eye(m)
        C = np.matmul(M, np.swapaxes(B, -2, -1)) + np.eye(m)

        # iterate over matrices and compute determinant
        infogain = np.zeros(game.k)
        for i, Ci in enumerate(C):
            L = cho_factor(Ci)[0]
            infogain[i] = 2 * np.sum(np.log(np.diag(L)))

        return infogain

class AsymptoticInfoGain(InfoGain):
    def __init__(self, correction='worst_case', eta=None, pareto_q=False, force_cell=False):
        self._user_learning_rate = eta is not None
        self._last_learning_rate = 1/eta if self._user_learning_rate else 1.
        self._correction = correction
        self._pareto_q = pareto_q
        self._force_cell=force_cell

    def info(self, gap_estimator : GapEstimator, verbose=False):
        game = gap_estimator.game
        # update learning rate
        if not self._user_learning_rate:
            min_norm = max(gap_estimator.minVnorm(),1e-10)
            self._last_learning_rate = min(self._last_learning_rate, 1 / np.sqrt(min_norm))
        eta = self._last_learning_rate
        # print(eta)
        lls = gap_estimator.lls
        # X = gap_estimator.X
        M = game.get_observation_maps().reshape(game.k, game.d)

        method = 'cell' if self._force_cell else None
        nu, V_norm = gap_estimator.alternatives(method=method)
        if verbose:
            print(nu, 'nu')
            print(lls.theta)

        if self._pareto_q:
            nu = nu[game.get_pareto_actions()]
            V_norm = V_norm[game.get_pareto_actions()]

        q = np.exp(- eta * V_norm)
        q[gap_estimator.winner()] = 0.
        if verbose:
            print(q, 'q')
            print(V_norm)
            print(eta)
        # compute sum_y q(y) (|<nu(y) - theta, x>| + beta_s^{1/2}\|x\|_{V_s^{-1}})^2 for each
        if self._correction == 'ucb':
            iucb, _ = gap_estimator.ucb()
            I_optimistic = np.zeros(gap_estimator.game.k)
            I_optimistic[iucb] = np.sqrt(lls.beta() * lls.var(M[iucb]))
        elif self._correction == 'worst_case':
            I_optimistic = np.sqrt(lls.beta() * lls.var(M))
        elif self._correction == 'none':
            I_optimistic  = 0.
        else:
            raise ValueError(f"Invalid choice {self._correction}")

        info = q @ (np.abs((nu - lls.theta) @ M.T) + I_optimistic) ** 2
        return info


class DirectedInfoGain(InfoGain):

    def info(self, gap_estimator : GapEstimator):
        winner = gap_estimator.winner()
        X_winner = gap_estimator.X[winner]
        means = gap_estimator.means()
        mean_winner = means[winner]
        cw = gap_estimator.lls.cw(gap_estimator.X - X_winner)
        plausible = mean_winner - means - cw <= 0
        if np.sum(plausible, dtype=int) == 1:
            # this check assumes that the optimal action is unique
            info = np.zeros(gap_estimator.game.k)
            info[plausible] = 1.
            return info

        cw_plausbile = cw[plausible]
        most_uncertain = np.argmax(cw_plausbile)
        w_t = gap_estimator.X[plausible][most_uncertain] - X_winner
        # game = gap_estimator.game
        # diff_X = difference_matrix(gap_estimator.X)
        # lower_bound = gap_estimator.lls.lcb(diff_X.reshape(-1, game.d)).reshape(game.k, game.k)
        # lower_bound = np.min(lower_bound, axis=1)
        # plausible = lower_bound <= 10e-10
        # P = game.X[plausible]
        #
        # PP = difference_matrix(P)
        # i, j = np.unravel_index(np.argmax(gap_estimator.lls.var(PP)), (len(P), len(P)))
        #
        # w_t = P[i] - P[j]

        # print(gap_estimator.lls.theta)
        return self._info(gap_estimator, w_t)

    def _info(self, gap_estimator, w):
        game = gap_estimator.game
        lls = gap_estimator.lls
        m = game.m
        d = game.d
        M = game.get_observation_maps()

        if m == 1:
            V_inv_w = cho_solve(lls.cholesky_factor, w)
            M = M.reshape(game.k, d)
            M_var = lls.var(M)
            nonzero_i = M_var > 1e-20  # avoid actions with zero norm, those have zero info gain
            info = np.zeros(game.k)
            M_V_inv_w = M[nonzero_i] @ V_inv_w
            w_V_inv_w = w @ V_inv_w
            info[nonzero_i] = np.log(w_V_inv_w) - np.log(w_V_inv_w - M_V_inv_w**2/(1 + M_var[nonzero_i]))
            return info

        # tentative update to V
        # VA = np.matmul(np.swapaxes(M, -2, -1), M) + lls.V

        # if w is 0-vector, return 0 info gain
        if np.sum(w * w) <= 10e-20:
            return np.ones(game.k)

        log_var_w = np.log(lls.var(w))
        log_var_wA = np.zeros(game.k)

        # compute log(var(w|A_i)) for each A_i
        for i, Mi in enumerate(M):
            VAi = lls.V + Mi.T @ Mi
            L = cho_factor(VAi)
            Vinv_w = cho_solve(L, w)
            log_var_wA[i] = np.log(w.dot(Vinv_w))

        info = log_var_w - log_var_wA
        return np.maximum(info, 0)

class UCBInfoGain(DirectedInfoGain):

    def info(self, gap_estimator):
        game = gap_estimator.game
        ucb_x = game.X[gap_estimator.ucb()[0]]
        return super()._info(gap_estimator, ucb_x)


class VarInfoGain(InfoGain):
    def __init__(self, num_samples=10000):
        super().__init__()
        self._num_samples = num_samples

        self._info1 = []
        self._info2 = []

    def info(self, gap_estimator):
        game = gap_estimator.game
        if game.m > 1:
            raise ValueError

        num_samples = self._num_samples
        pareto_X = game.get_actions()[game.get_pareto_actions()]
        samples = gap_estimator.lls.posterior_samples(num_samples)
        mean_theta = np.mean(samples, axis=0)

        amax = np.argmax(pareto_X.dot(samples.T), axis=0)
        Z = np.zeros((num_samples, len(pareto_X)), dtype=bool)
        Z[np.arange(num_samples), amax] = 1

        q = np.sum(Z, axis=0)

        # check if all samples are on the same action
        if np.any(q == self._num_samples):
            return q/self._num_samples

        q = q / self._num_samples

        cond_mean_theta = np.zeros((len(pareto_X), game.d))
        for i, mask_i in enumerate(Z.T):
            if q[i] > 0:
                # print(mask_i)
                cond_mean_theta[i] = np.mean(samples[mask_i], axis=0)

        M = game.get_observation_maps().reshape(game.k, game.d)
        return q @ ((cond_mean_theta - mean_theta) @ M.T)**2


class LaplaceMIInfoGain(InfoGain):

    def info(self, gap_estimator: GapEstimator):
        game = gap_estimator.game
        lls = gap_estimator.lls
        pareto_A = game.get_pareto_actions()
        pareto_X = game.get_actions()[pareto_A]
        best_x_ind = np.argmax(lls.mean(pareto_X))
        nu, norm = gap_estimator.alternatives()

        nu = nu[pareto_A]
        norm = norm[pareto_A]

        M = game.get_observation_maps().reshape(game.k, game.d)[pareto_A]

        N = (2 * np.pi) ** (-game.d / 2) * np.prod(np.diag(lls.cholesky_factor[0]))
        # compute Laplace approximation with correct normalization
        q = np.exp(- 0.5 * norm) / np.sqrt(norm) / 2 / N
        q[best_x_ind] = 0.
        q[best_x_ind] = 1 - np.sum(q)

        q_cond = q * np.exp(-0.5 * (np.square((nu - lls.theta) @ M.T)))
        q_cond[:, best_x_ind] = 0
        q_cond[:, best_x_ind] = 1 - np.sum(q_cond, axis=1)
        # # note, we ignore first coordinate of q for numerical stability
        # return - np.sum(q[1:] * np.log(q[1:])) + np.sum(q_cond[:,1:] * np.log(q_cond[:,1:]), axis=1)
        return - np.sum(q * np.log(q)) + np.sum(q_cond * np.log(q_cond), axis=1)


class SampleMIInfoGain(InfoGain):
    def __init__(self, num_samples=100000):
        super().__init__()
        self._num_samples = num_samples

        self._info1 = []
        self._info2 = []

    def info(self, gap_estimator):
        game = gap_estimator.game
        if game.m > 1:
            raise ValueError
        lls = gap_estimator.lls
        pareto_X = game.get_actions()[game.get_pareto_actions()]

        # prior entropy
        entropy = sample_optimal_action(lls.theta, lls.V, pareto_X, self._num_samples, lls.cholesky_factor)

        # conditional entropy
        M = game.get_observation_maps()

        # iterate over all actions
        cond_entropy = np.zeros(game.k)
        for i, Mi in enumerate(M):
            Mi = Mi[0]
            cond_entropy[i] = sample_optimal_action(lls.theta, lls.V + np.outer(Mi, Mi), pareto_X, num_samples=self._num_samples)

        # info =  np.maximum(entropy - cond_entropy, 0)
        info = entropy - cond_entropy
        return info

def sample_optimal_action(theta, V, pareto_X, num_samples, cholesky_factor=None):
    if cholesky_factor is None:
        cholesky_factor = cho_factor(V)
    d = theta.shape[-1]
    eta = np.random.normal(0, 1, d * num_samples).reshape(num_samples, d)
    samples = theta + solve_triangular(cholesky_factor[0], eta.T).T


    # prior entropy
    amax = np.argmax(pareto_X.dot(samples.T), axis=0)
    Z = np.zeros((num_samples, len(pareto_X)))
    Z[np.arange(num_samples), amax] = 1
    q = np.sum(Z, axis=0) / num_samples

    mask = q > 0
    return -np.inner(q[mask], np.log(q[mask]))
