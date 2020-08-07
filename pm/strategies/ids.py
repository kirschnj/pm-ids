from pm.strategy import Strategy
import numpy as np
from scipy.linalg import cho_solve, cho_factor

from pm.utils import difference_matrix, compute_nu





def full(indices, game, estimator, q, nu):
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


    lower_bound = estimator.regret_lower_2(indices)
    return infogain

def directeducb(indices, game, estimator, q, nu):
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

def directed2(indices, game, estimator, q, nu):
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
    lower_bound = estimator.regret_lower_2(I)
    return I[lower_bound <= 10e-10]


def directed3(indices, game, estimator, q, nu):
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


def info_game(indices, game, estimator, q, nu):
    delta = 1. / estimator.lls._t

    I = np.zeros(len(indices)) #to be vectorized
    k = len(indices)

    # compute the main info term

    diff = np.zeros((k,k))
    for i in indices:
        for j in indices:
            diff[i,j] = np.dot(nu[j,:] - estimator.lls._theta, game._X[i,:])**2
    for i in indices:
        I[i] = np.dot(q, diff[i,:].T)

    # compute the additional vanishing info term

    J = np.zeros(len(indices)) #to be implemented
    # compute plausible maximizer set
    ucb = estimator.ucb(indices)
    winner = np.argmax(ucb)
    w = game.get_actions(winner)
    epsilon = ucb[winner] - np.dot(estimator.lls._theta, game._X[winner,:])


    d = game.get_d()

    A = game.get_actions(indices)
    B = cho_solve(estimator.lls.get_cholesky_factor(), A.T).T
    # multiply cholesky factors and add unit matrix to get: A^\T V^{-1} A + eye(m)
    C = np.matmul(A, np.swapaxes(B, -2, -1))

    for i in indices:
        if i != winner:
            Xi = game._X[i,:]
            # obtain V_t^{-1}X_i
            Vinv_Xi = cho_solve(estimator.lls.get_cholesky_factor(), Xi.T)
            J[i] = epsilon * estimator._lls.beta(delta) * (np.dot(Xi,Vinv_Xi))

    infogain = I + J

    return infogain

class IDS(Strategy):

    def __init__(self, game, estimator, infogain, deterministic=False, anytime=True):
        super().__init__(game, estimator)
        self._infogain = infogain
        self._deterministic = deterministic
        self.anytime = anytime
        if self.anytime:
            self.update= True

    def get_next_action(self):
        if self._deterministic:
            return self._dids()
        else:
            if self.anytime:
                return self._anytime_ids()
            else:
                return self._ids()

    def add_observations(self,indices,y):
        if not self.anytime:
            super().add_observations(indices,y)

        else:

            if self.update: #explore and collect data
                super().add_observations(indices,y)
            else: # online increase global time count
                self._estimator.lls._t +=1

    def _mixed_ratio(self, A, B, C, D, p_new, ratio, p):

        # invalid x, return previous ratio
        if p_new < 0 or p_new > 1:
            return ratio, p

        # if the ratio is better with x, return new ratio and x
        tmp_ratio = (p_new * A + (1 - p_new) * B) ** 2 / (p_new * C + (1 - p_new) * D)
        if tmp_ratio < ratio:
            return tmp_ratio, p_new

        return ratio, p

    def oco(self,indices):
        """
        Compute the q-learner, which is a distribution over the constraints

        outputs:
        q : K vector containing the constraint mixing for each action (0 for the ucb one)
        """

        delta = 1/(self._estimator.lls._t * np.log(self._estimator.lls._t+2)) #= 1/(tlog(t)) adding +1 for numerical issues at initialization
        eta = 1./ np.sqrt(self._estimator._lls.beta(delta)) #=> 1/\sqrt{\beta_t}
        theta_hat = self._estimator._lls._theta
        V = self._estimator._lls._V
        k = len(indices)
        q = np.zeros(k)

        nu = compute_nu(self._estimator._lls, self._game)
        # ucb = self._estimator.ucb(indices)
        # winner = np.argmax(ucb) # former def of winner

        actions = self._game.get_actions(indices)
        means = self._estimator._lls.mean(actions)
        winner = np.argmax(means) #empirical best

        w = actions[winner]

        for i in indices:
            if i != winner:
                D = nu[i,:] - theta_hat
                V_norm = np.matmul(D, np.matmul(V,D))
                q[i] = np.exp(-eta * (V_norm))

        # normalize
        q /= np.linalg.norm(q)
        return q , nu

    def _ids(self):
        """
        Compute the randomized IDS solution
        https://www.wolframalpha.com/input/?i=d%2Fdx+(Ax+%2B+(1-x)*B)^2%2F(Cx+%2B+(1-x)D)+
        """
        indices = self._game.get_indices()
        # indices = plausible_maximizers_2(self._game, self._estimator)
        regret = self._estimator.regret_upper(indices)

        q, nu  = self.oco(indices)

        infogain = self._infogain(indices, self._game, self._estimator, q, nu) # need to change template for other infogain functions

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

    def _dids(self):
        """
        Compute the deterministic IDS solution
        """
        indices = self._game.get_indices()
        # regret = self._estimator.regret_upper(indices)
        #
        infogain = self._infogain(indices, self._game, self._estimator)

        # print(full(indices, self._game, self._estimator) - directed2(indices, self._game, self._estimator))

        regret = np.max(self._estimator.ucb(indices)) - self._estimator.lcb(indices)
        ratio = regret**2/infogain
        return indices[np.argmin(ratio)]

    def _anytime_ids(self,b=2):
        """
        Compute the IDS solution when there's "enough" data,
        and otherwise fallback on UCB
        """
        #
        # self.s = 1
        t = self._estimator.lls._t
        # print(t)
        delta_t = 1/(t * np.log(t+2)) # adding +1 to avoid numerical issues at initialization
        indices = self._game.get_indices()
        q, nu  = self.oco(indices)
        actions = self._game.get_actions(indices)
        means = self._estimator._lls.mean(actions)
        winner = np.argmax(means) #empirical best



        theta_hat = self._estimator.lls.theta()
        V = self._estimator.lls._V
        info_t = [np.matmul(nu[i,:] - theta_hat, np.matmul(V,nu[i,:] - theta_hat)) if i != winner else 10**6 for i in indices ]
        # exploration condition
        # print(min(info_t))
        # print(self._estimator.lls.beta(delta_t))
        if np.min(info_t) < self._estimator.lls.beta(delta_t ) :
            self.update = True #exploration => collect data

            #compute beta_s
            # accessing s through the Trace of V_s: is it correct ?
            s = np.sum(np.diag(self._estimator.lls.get_cholesky_factor()[0]))
            beta_s = self._estimator.lls.beta(1/(s**b+1))

            ucbs = self._estimator.lls.ucb(actions, 1/s**b)
            gaps = ucbs - means[winner]
            #compute delta_s
            delta_s = gaps[winner]

            #check the UCB fallback condition delta_s < gaps/2 for all non-winners
            if np.min([gaps[y]-(2*delta_s) if y != winner else 1 for y in indices ])>0 :
                return self._ids()
            else:
                return indices[np.argmax(ucbs)]



        else:
            self.update=False
            return winner






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
