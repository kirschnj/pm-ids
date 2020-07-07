from pm.strategy import Strategy
import numpy as np
from scipy.linalg import cho_solve, cho_factor
from pm.utils import compute_nu



#TODO: implement the OCO learner





# def compute_nu(estimator, game):
#     """
#     Compute the alternative parameter for each cell but that of the empirical best arm.
#     """
#     indices = game._I
#     d = game._d
#     X = game.get_actions(indices)
#     theta = estimator._theta
#     V = estimator._V
#     nu = np.zeros((len(indices),d))
#     # for each action, solve the quadratic program to find the alternative
#     for i in indices:
#         Xi = X[i,:]
#         Ci = game.get_cell_constraints(i)
#         x = cp.Variable(game._d)
#         prob = cp.Problem(cp.Minimize(cp.quad_form(x, V)), [Ci @ x <= (-Ci @ theta)])
#         prob.solve()
#
#         nu[i:] = x.value #+ theta # we solved for x=nu-hat_theta
#
#     #normalize as per our unit ball hypothesis
#     nu /= np.linalg.norm(nu, axis=1)[:, None]
#     return nu

#TODO : write the test


# TODO: create a new infogain that computes I_t + J_t as defined in Eq 103-105

def info_game(indices, game, estimator, q, nu):
    delta = 0.05 # this should be 1/t but I don't know yet how to do this
    I = np.zeros(len(indices)) #to be vectorized
    k = len(indices)

    # compute the main info term

    diff = np.zeros((k,k))
    for i in indices:
        for j in indices:
            diff[i,j] = np.dot(nu[j,:] - game._theta, game._X[i,:])**2
    for i in indices:
        I[i] = np.dot(q[i,:], diff)

    # compute the additional vanishing info term

    J = np.zeros(len(indices)) #to be implemented
    # compute plausible maximizer set
    ucb = estimator.ucb(indices)
    winner = np.argmax(ucb)
    w = game.get_actions(winner)
    epsilon = ucb[winner] - game._theta


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
            J[i] = epsilon * estimator.beta(delta) * (np.matmul(Xi,V_inv_Xi))

    infogain = I + J

    return infogain


class OPTIDS(Strategy):

    def __init__(self, game, estimator, infogain):
        super().__init__(game, estimator)
        self._infogain = infogain
        self.eta = 1  #not sure it is defined


    # def add_observations(self, indices, y):
    #     self._estimator.add_data(indices, y)

    def oco(self,indices):
        """
        Compute the q-learner, which is a distribution over the constraints

        outputs:
        q : K vector containing the constraint mixing for each action (0 for the ucb one)
        """
        eta = self.eta#?
        delta = 0.01
        theta_hat = self._estimator._lls._theta
        V = self._estimator._lls._V
        k = len(indices)
        q = np.zeros(k)

        nu = compute_nu(self._estimator._lls, self._game)
        ucb = self._estimator.ucb(indices)
        winner = np.argmax(ucb)
        w = self._game.get_actions(winner)

        for i in indices:
            if i != winner:
                D = nu[i,:] - theta_hat
                V_norm = np.matmul(D, np.matmul(V,D))
                q[i] = np.exp(-eta * (V_norm))

        # normalize
        q /= np.linalg.norm(q)
        return q

    def _optids(self):
        """
        Compute the randomized IDS solution
        """

        indices = self._game.get_indices()
        # indices = plausible_maximizers_2(self._game, self._estimator)
        regret = self._estimator.regret_upper(indices)
        q = self.oco(indices)


        infogain = self._infogain(indices, self._game, self._estimator,q, nu)

        # Next is similar to other ids, I need to figure out how to not duplicate code.

    def get_next_action(self):
        # update beta
        # self.beta = 2*(1+ 1/np.log(t))*np.log(t) + 3*self.game._d*np.log(np.log(t))
        return self._optids()

    def id(self):
        return "opt-ids"
