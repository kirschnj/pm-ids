from pm.strategy import Strategy
import numpy as np
from scipy.linalg import cho_solve, cho_factor

from pm.utils import difference_matrix, psd_norm_squared
import cvxpy as cp


class Solid(Strategy):
    def __init__(self, game, estimator, lambda_1=0., z_0=30, alpha_l=0.1, alpha_w = 0.5, lambda_max=10):
        #check default values of parameters
        super().__init__(game, estimator)
        self.lambda_t = lambda_1
        self.lambda_1 = lambda_1 # for resets at phase ends
        self.lambda_max = lambda_max
        self.z_t = z_0
        self.k = 0 #phase counter
        self.p_k = z_0 # z_k exp(2*k)
        self.z_k = z_0 # z_0 exp(k)
        self.alpha_w = alpha_w
        self.alpha_l = alpha_l
        self.explo_rounds = 0 # exploration counter per phase
        self.phase_length = 0
        self.phase = 1 # K_1 in article

        # This is only for the one-context-bandit game that is available now:
        self.K = len(self._game.get_indices())
        self.w_t = np.ones(self.K) / self.K #uniform over actions
        #for multi context bandit we need a matrix (nb_context x nb_actions)
        self._t = 1



    def compute_alt(self, indices, V_matrix, theta=None):
        """
        Generic function of the compute_nu from ids for different V_matrices
        Compute the alternative parameters and V_matrix-norm (squared) for each cell
        and returns the minimum norm and the corresponding parameter vector.
        """
        d = self._game.d
        X = self._game.get_actions(indices)
        if theta == None:
            theta = self._estimator.lls.theta
        else :
            theta = theta.copy()
        nu = np.zeros((len(indices), d))
        C = difference_matrix(X)
        # for each action, solve the quadratic program to find the alternative
        for i in indices:
            x = cp.Variable(d)
            q = -2 * (V_matrix @ theta)
            G = -C[i, :, :]

            prob = cp.Problem(cp.Minimize(cp.quad_form(x, V_matrix) + q.T @ x), [G @ x <= 0])
            prob.solve()

            nu[i,:] = x.value

        # check corner cases in the bandit case : can the projected nu have a very large norm ? => regularization ?
        # normalize as per our unit ball hypothesis => creates bugs when the projection on the cone is too close to origin. Also does it make sense ?
        # nu /= np.linalg.norm(nu, axis=1)[:, None]
        V_norm = psd_norm_squared(nu - theta, V_matrix)

        #recompute at every round because estimator changes
        means = self._estimator.lls.mean(self._game.get_actions(indices))
        self._winner = np.argmax(means)


        V_norm_tmp = V_norm.copy()

        # compute minimum V-norm without winner
        V_norm_tmp[self._winner] = np.Infinity
        min_V_norm = np.min(V_norm_tmp)
        ind_min = np.argmin(V_norm_tmp)

        return nu[ind_min,:], min_V_norm

    def compute_q(self):
        indices = self._game.get_indices()
        X = self._game.get_actions(indices)

        # compute df_t
        delta_f_t = self._estimator.ucb(indices, delta=1/self.explo_rounds)

        #compute dg_t

        # weighted matrix \sum_a w_a aa^T
        # print(self.w_t[0]*np.outer(X[0,:],X[0,:]))
        Vw = self.w_t[0] * np.outer(X[0,:],X[0,:])
        for a in indices[1:]:
            Vw = np.add(Vw, self.w_t[a] * np.outer(X[a,:],X[a,:]) )


        # print('Vw is '+ str(Vw))

        theta_alt, W_norm_min = self.compute_alt(indices, Vw)

        gaps_sq = (X @ (self._estimator.lls.theta - theta_alt)) ** 2

        delta_g_t = gaps_sq + np.sqrt(self._estimator.lls.beta(1/self.explo_rounds) * self._estimator.var(indices))

        return delta_f_t + self.lambda_t * delta_g_t, theta_alt, W_norm_min



    def update_alphas(self):
        return 1/np.sqrt(self.p_k), 1/np.sqrt(self.p_k)

    def update_w_l(self, indices):
        X = self._game.get_actions(indices)
        #compute q_t(x,a)
        q_t, theta_alt, W_norm_min = self.compute_q()

        softmax = np.exp(self.alpha_w * q_t)

        self.w_t = np.multiply(self.w_t,softmax)
        self.w_t /= np.sum(self.w_t)

        g_t = W_norm_min + np.dot(self.w_t, np.sqrt(self._estimator.lls.beta(1/self.explo_rounds) * self._estimator.var(indices))) - 1/self.z_k
        self.lambda_t = np.min([self.lambda_t - self.alpha_l*g_t, self.lambda_max])






    def get_next_action(self):
        indices = self._game.get_indices()
        #same as in ids, but article takes delta=1/n and uses n instead of logdet(V_t)
        beta_t = self._estimator.lls.beta(1/(self._t * np.log(self._t + 2)))  # adding +1 to avoid numerical issues at initialization


        theta_min, min_V_norm = self.compute_alt(indices, self._estimator.lls.V)

        self._min_V_norm = min_V_norm

        if self._min_V_norm > beta_t:
            #exploitation step

            return winner

        else:
            chosen_action = np.random.choice(indices, p=self.w_t)
            # updates
            self.explo_rounds += 1
            self.phase_length += 1
            self.update_w_l(indices)

            if self.phase_length == self.p_k:
                #update phase counters
                self.phase += 1
                self.phase_length = 0
                self.z_k *= np.exp(self.phase / (self.phase -1 ))
                self.p_k = self.z_k * np.exp(2*self.phase)
                # updates alphas
                self.alpha_w , self.alpha_l = self.update_alphas()
                #reset
                self.w_t =  np.ones(self.K) / self.K #uniform over actions #uniform over actions
                self.lambda_t = self.lambda_1
            return chosen_action

    def add_observations(self, indices, y):

        super().add_observations(indices, y)

        self._t += 1  # increase step counter each time we get the data

    def id(self):
        return "Solid"
