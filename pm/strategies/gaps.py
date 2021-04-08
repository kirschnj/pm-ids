import numpy as np
from scipy.linalg import cho_solve, cho_factor
import osqp
from scipy import sparse

from pm.game import Game
from pm.strategies.lls import RegularizedLeastSquares
from pm.utils import difference_matrix, psd_norm_squared


class GapEstimator:
    def __init__(self, alternatives='halfspace', truncate=None, integrate=False):
        self.t = 0
        self._nu_warm_start = None
        self._alternatives_method = alternatives
        self._truncate = truncate
        self._integrate = integrate
        self.reset()

    def estimate(self, game : Game, lls : RegularizedLeastSquares):
        self.reset()
        self.game = game
        self.lls = lls
        self.t += 1
        self.X = self.game.X

    def exploit(self):
        return self.minVnorm() >= self.beta_t()

    def alternatives(self, method=None):
        if method is None:
            method = self._alternatives_method
        if not method in self._alternatives:
            if method == 'halfspace':
                self._alternatives[method], self._V_norm[method] = self._alternatives_halfspace()
            elif method == 'cell':
                self._alternatives[method], self._V_norm[method] = self._alternatives_cell()
            else:
                raise RuntimeError("Invalid Option")
        return self._alternatives[method], self._V_norm[method]

    def _alternatives_cell(self):
        """
               Compute the alternative parameters and Vnorm for each cell
               """
        d = self.game.d
        X = self.X
        theta = self.lls.theta
        V = self.lls.V
        nu = np.zeros((self.game.k, d))
        C = difference_matrix(X)
        # for each action, solve the quadratic program to find the alternative
        for i in range(self.game.k):
            if i == self.winner():
                nu[i] = theta
                continue

            q = -2 * (V @ theta)
            G = -C[i, :, :]
            P = 2 * V

            sparse_G = sparse.csc_matrix(G)
            sparse_P = sparse.csc_matrix(P)

            settings = dict(
                verbose=False,
                eps_abs=1e-6,
                eps_rel=1e-6,
            )
            # if self._osqp_solvers[i] is None:
            prob = osqp.OSQP()
            prob.setup(P=sparse_P, q=q, A=sparse_G, u=np.zeros(G.shape[0]), **settings)
            # storing the problem instances and updating the matrices doesn't seem to work
            # self._osqp_solvers[i] = prob
            # else:
            #     prob = self._osqp_solvers[i]
            #     prob.update(Px=sparse.triu(sparse_P).data, Ax=sparse_G.data)
            #     prob.update(q=q)#, Ax=G, Ax_idx=sparse_G.indices)

            if self._nu_warm_start is not None:
                prob.warm_start(x=self._nu_warm_start[i])

            res = prob.solve()
            # print(res.x - nu[i])
            nu[i] = res.x

        # check corner cases in the bandit case : can the projected nu have a very large norm ? => regularization ?
        # normalize as per our unit ball hypothesis => creates bugs when the projection on the cone is too close to origin. Also does it make sense ?
        # nu /= np.linalg.norm(nu, axis=1)[:, None]
        V_norm = psd_norm_squared(nu - theta, V)
        self._nu_warm_start = nu
        return nu, V_norm

    def _alternatives_halfspace(self):
        means = self.means()
        winner = self.winner()
        # x_best = self.X[winner]
        x_diff = self.X - self.X[winner]
        gaps = means[winner] - means
        theta = self.lls.theta

        X_norm = self.lls.var(x_diff)
        X_norm[winner] = 1.  # avoid division by zero
        nu = theta - (gaps / X_norm * cho_solve(self.lls.cholesky_factor, x_diff.T)).T
        V_norm = gaps ** 2 / X_norm
        return nu, V_norm

    def winner(self, mean=False):
        if self._winner is None:
            if mean:
                # if self.context is not None and self._integrate:
                #     self._winner = np.argmax(self.means(), axis=1)
                # else:
                self._winner = np.argmax(self.means())
            else:
                self._winner = np.argmin(self.gaps())
        return self._winner

    def means(self):
        if self._means is None:
            self._means = self.lls.mean(self.X)

        return self._means

    def gaps(self, context=None):
        if context is not None:
            self.reset()
            self.X = self.game.X[context]

        if self._gaps is None:
            self._gaps = self._compute_gaps()
            if self._truncate is not None:
                self._gaps = np.minimum(self._gaps, self._truncate)
        return self._gaps

    def _compute_gaps(self):
        raise NotImplemented

    def minVnorm(self):
        if self._ms is None:
            V_norm_tmp = self.alternatives()[1].copy()
            # compute minimum V-norm without winner
            V_norm_tmp[self.winner()] = np.Infinity
            self._ms = np.min(V_norm_tmp)
        return self._ms

    def beta_t(self):
        if self._beta_t is None:
            _t = max(2, self.t)
            self._beta_t = self.lls.beta(1/(_t * np.log(_t)))
        return self._beta_t

    def ucb(self):
        if self._ucb_i is None:
            ucb = self.lls.ucb(self.X)
            self._ucb_i = np.argmax(ucb)
            self._ucb_val = ucb[self._ucb_i]

        return self._ucb_i, self._ucb_val

    def reset(self):
        self._alternatives = dict()
        self._V_norm = dict()
        self._winner = None
        self._means = None
        self._ms = None
        self._beta_t = None
        self._gaps = None
        self._ucb_i = None
        self._ucb_val = None

class ValueGap(GapEstimator):
    def _compute_gaps(self):
        lcb, ucb = self.lls.lcb_ucb(self.X)
        return np.max(ucb) - lcb

class FastValueGap(GapEstimator):
    def _compute_gaps(self):
        ucb = self.lls.ucb(self.X)
        return np.max(ucb) - self.means()

class DiffGap(GapEstimator):
    def _compute_gaps(self):
        D = difference_matrix(self.X)
        # compute ucb score for all differences and max out columns
        return np.max(self.lls.ucb(D), axis=0)

class FastDiffGap(GapEstimator):
    def _compute_gaps(self):
        # if self.context is not None and self._integrate:
        #     winners = self.X[self.winner(mean=True)]
        #     ucb_diff = 0
        #     # we compute a gap estimate that is relaxed with Jensen
        #     for c, p, w in zip(range(self.game.num_contexts), self.game.cdistr, winners):
        #         Xc = self.X[c]
        #         means = self.means()[c]
        #         ucb_diff += p * np.max(means + self.lls.cw(Xc - Xc[w]))
        # else:
        x_best = self.X[self.winner(mean=True)]
        ucb_diff = np.max(self.means() + self.lls.cw(self.X - x_best))
        return ucb_diff - self.means()

class BayesianGap(GapEstimator):

    def __init__(self, alternatives='halfspace', truncate=None, num_samples=10000):
        super().__init__(alternatives=alternatives, truncate=truncate)
        self._num_samples = num_samples

    def _compute_gaps(self):
        samples = self.lls.posterior_samples(self._num_samples)
        mean = np.mean(samples, axis=0)  # compute mean from samples to prevent negative gaps
        # print(self.X.dot(mean).shape, np.max(self.X.dot(samples.T), axis=0).shape)
        pareto_X = self.X[self.game.get_pareto_actions()]
        return np.mean(np.max(pareto_X.dot(samples.T), axis=0)) - self.X.dot(mean)

        # print("")
        # print(gaps)
        # # ucb = self.lls.ucb(self.X)
        #
        # print(np.max(self.means()) - self.means())
        # return gaps
