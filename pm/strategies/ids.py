import logging
import numpy as np
import cvxpy as cp
from scipy.linalg import cho_solve, cho_factor
import osqp
from scipy import sparse

from pm.utils import difference_matrix, psd_norm_squared
from pm.strategy import Strategy


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

    def __init__(self, game, estimator, infogain, deterministic=False, fast_ratio=False):
        """
        fast_ratio=True: compute minimizing distribution only between empirical maximizer + one other action
        """
        super().__init__(game, estimator)
        self._infogain = infogain
        self._deterministic = deterministic
        self._update_estimator = True
        self._t = 1  # step counter
        self._fast_ratio = fast_ratio

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

    def _two_action_ratio(self, D1, D2, I1, I2):
        """
        computes optimal trade-off between two actions
        p_min = argmin_p ((1-p)*D1 + p*D2)**2/((1-p)*I1 + p*I2)
        returns p_min, Psi(p_min)
        """

        # if info gain is zero for both action, randomize uniformly
        if I1 == 0. and I2 == 0.:
            return 0.5, np.inf

        # make sure that D1 <= D2 by flipping the actions if necessary
        if D2 < D1:
            D1, D2 = D2, D1
            I1, I2 = I2, I1
            flip = True
        else:
            flip = False

        if I1 >= I2:
            p = 0.
            ratio = D1**2/I1
        elif D1 == D2: # I1 < I2
            p = 1.
            ratio = D2**2/I2
        else:
            p = D1/(D2 - D1) - 2*I1/(I2 - I1)
            p = max(0., min(1., p))  # clip to [0,1]
            ratio = ((1-p)*D1 + p*D2)**2/((1-p)*I1 + p*I2)

        if flip:
            p = 1. - p

        return p, ratio

    def _ids_sample(self, indices, gaps, infogain):
        """
        Compute the randomized IDS solution
        https://www.wolframalpha.com/input/?i=d%2Fdx+(Ax+%2B+(1-x)*B)^2%2F(Cx+%2B+(1-x)D)+
        """
        if self._fast_ratio:
            winner = np.argmin(gaps)

        best_p = None
        best_i = None
        best_j = None
        best_ratio = np.inf

        for j in range(len(indices)):
            # if the fast_ratio flag is set, compute the ratio only between j and the winner
            if self._fast_ratio:
                inner_range = [winner]
            else:
                inner_range = range(j+1,len(indices))

            for i in inner_range:

                D1, D2 = gaps[i], gaps[j]
                I1, I2 = infogain[i], infogain[j]

                p, ratio = self._two_action_ratio(D1, D2, I1, I2)

                if ratio < best_ratio:
                    best_ratio = ratio
                    best_i, best_j = i, j
                    best_p = p

        # p=1 means we pick j
        if np.random.binomial(1, p=best_p):
            return indices[best_j]
        else:
            return indices[best_i]

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

    def __init__(self, game, estimator, fast_ratio=False, lower_bound_gap=False, opt2=False, alpha=1., ucb_switch=False,fast_info=False):
        self.lower_bound_gap = lower_bound_gap
        self.mms = 1.
        self.opt2 = opt2
        self.alpha = alpha
        self.fast_info = fast_info
        self.ucb_switch = ucb_switch
        self._nu = None
        self._osqp_solvers = [None] * game.k
        super().__init__(game, estimator, infogain=None, deterministic=False, fast_ratio=fast_ratio)

    def _info_game(self, indices, winner, nu, beta, V_norm, alpha=None):
        # compute mixing weights
        eta = 1. / np.sqrt(self.mms)
        q = np.exp(- eta * V_norm)
        q[winner] = 0.
        iucb = np.argmax(self._estimator.ucb(indices))
        if alpha is None:
            alpha = self.alpha

        # compute info gain
        X = self._game.get_actions(indices)
        # TODO: include optimistic terms
        lls = self._estimator.lls
        # compute sum_y q(y) (|<nu(y) - theta, x>| + beta_s^{1/2}\|x\|_{V_s^{-1}})^2 for each x

        # Johannes 20.1.2021:
        # the optimistic part of the information gain is to make asymptotic information gain robust
        # in the case when the estimates are inaccurate. Also this parts guarantees the worst-case bounds.

        # Option 1, from the paper:
        # I_optimistic = np.sqrt(lls.beta()*lls.var(X))

        # Option 2, focus on UCB action in finite time regime:
        # This one improves performance in finite time, but pushes the regime switch, it seems
        if self.opt2:
            I_optimistic = np.zeros(len(indices))
            I_optimistic[iucb] = np.sqrt(lls.beta()*lls.var(X[iucb]))
        else:
            I_optimistic = np.sqrt(lls.beta()*lls.var(X))

        I = q @ (np.abs((nu - lls.theta) @ X.T) + I_optimistic)**2
        return I

    def compute_nu(self, indices, winner, nu_old):
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
            if i == winner:
                nu[i] = theta
                continue

            x = cp.Variable(d)

            if nu_old is not None:
                x.value = nu_old[i]
            q = -2 * (V @ theta)
            G = -C[i, :, :]

            prob = cp.Problem(cp.Minimize(cp.quad_form(x, V) + q.T @ x), [G @ x <= 0])
            # prob.solve(solver='ECOS')#, solver_specific_opts={'max_iter' : 2000})
            prob.solve(solver='OSQP', adaptive_rho=True, max_iter=2000, warm_start=True)
            # Solve problem
            nu[i] = x.value



        # check corner cases in the bandit case : can the projected nu have a very large norm ? => regularization ?
        # normalize as per our unit ball hypothesis => creates bugs when the projection on the cone is too close to origin. Also does it make sense ?
        # nu /= np.linalg.norm(nu, axis=1)[:, None]
        V_norm = psd_norm_squared(nu - theta, V)
        return nu, V_norm

    def compute_nu_osqp(self, indices, winner, nu_old):
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
            if i == winner:
                nu[i] = theta
                continue

            q = -2 * (V @ theta)
            G = -C[i, :, :]
            P = 2*V

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

            if nu_old is not None:
                prob.warm_start(x=nu_old[i])

            res = prob.solve()
            # print(res.x - nu[i])
            nu[i] = res.x

        # check corner cases in the bandit case : can the projected nu have a very large norm ? => regularization ?
        # normalize as per our unit ball hypothesis => creates bugs when the projection on the cone is too close to origin. Also does it make sense ?
        # nu /= np.linalg.norm(nu, axis=1)[:, None]
        V_norm = psd_norm_squared(nu - theta, V)
        return nu, V_norm

    def compute_nu_fast(self, indices, means, winner):
        x_best = self._game.get_actions(winner)
        X = self._game.get_actions(indices)
        gaps = means[winner] - means
        theta = self._estimator.lls.theta

        X_norm = self._estimator.lls.var(X - x_best)
        X_norm[winner] = 1.  # avoid division by zero
        nu = theta - (gaps/X_norm * cho_solve(self._estimator.lls.get_cholesky_factor(),  -(X - x_best).T)).T
        V_norm = gaps**2/X_norm
        return nu, V_norm

    def get_next_action(self):
        """
        Compute the IDS solution when there's "enough" data.
        """
        indices = self._game.get_indices()
        #we may want to try other values for beta_t
        _t = max(2, self._t)
        beta_t = self._estimator.lls.beta(1/(_t * np.log(_t)))

        # only re-compute quantities when the estimator changes
        if self._update_estimator:
            # compute winner
            means = self._estimator.lls.mean(self._game.get_actions(indices))
            self._means = means
            self._winner = np.argmax(means)

            # compute alternatives and V_norm
            if self.fast_info:
                self._nu, self._V_norm = self.compute_nu_fast(indices, means, self._winner)
            else:
                # self._nu, self._V_norm = self.compute_nu(indices, self._winner, nu_old=self._nu)
                self._nu, self._V_norm = self.compute_nu_osqp(indices, self._winner, nu_old=self._nu)
                # print(nu - self._nu)
                # print(np.linalg.norm(nu - self._nu))
            V_norm_tmp = self._V_norm.copy()

            # compute minimum V-norm without winner
            V_norm_tmp[self._winner] = np.Infinity
            self.ms = np.min(V_norm_tmp)
            # maximum observed distance to closest parameter, used in learning rate
            self.mms = np.max([self.ms, self.mms])

        winner, V_norm, nu, means = self._winner, self._V_norm, self._nu, self._means

        # check exploration/exploitation condition
        if self.ms < beta_t:
            self._update_estimator = True  # exploration => collect data

            # alternative way to compute ms
            gaps = means[winner] - means
            gaps[winner] = np.inf
            second_best = np.argmin(gaps)
            x_best = self._game.get_actions(winner)
            x_2best = self._game.get_actions(second_best)
            alt_ms = gaps[second_best]**2/self._estimator.lls.var(x_best - x_2best)

            if self.ucb_switch:
                gap_2best = gaps[second_best]
                ucb_score = self._estimator.ucb(indices)
                ucb_i = np.argmax(ucb_score)
                delta = ucb_score[ucb_i] - means[winner]
                if delta > gap_2best:
                    logging.debug(f"delta={delta}, Delta_min={gap_2best}")
                    logging.debug("Choosing UCB action")
                    return ucb_i
                logging.debug("choosing IDS action")


            logging.debug(f"m_s={self.ms:0.3f}, m_s_alt={alt_ms:0.3f}, beta_t={beta_t:0.3f}, t={self._t}, log(t)={np.log(_t):0.3f}")

            gaps = self._estimator.gap_upper(indices)
            delta_s = gaps[winner]

            # Johannes (20.01.2021): This option needs to be explored because of a corner case in the analysis
            if self.lower_bound_gap:
                gaps += np.max(1/np.sqrt(self._estimator.lls.s) - delta_s, 0)

            if delta_s < 1/np.sqrt(self._estimator.lls.s):
                logging.info("minimum gap too small?")

            alpha = None
            if self.ucb_switch:
                alpha = 0.
            infogain = self._info_game(indices, winner, nu, beta_t, V_norm, alpha=alpha)
            return self._ids_sample(indices, gaps, infogain)
        else:
            logging.info("exploit")
            self._update_estimator = False
            return winner
