from pm.strategy import Strategy
import numpy as np
from scipy.linalg import cho_solve, cho_factor

from pm.utils import difference_matrix, psd_norm_squared
import cvxpy as cp


class Solid(Strategy):
    def __init__(self, game, estimator, lambda_1=0., z_0=100, alpha_l=0.1, alpha_w = 0.5, lambda_max=10, reset=True):
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
        self.reset = reset

    def compute_nu(self, indices):
        """
        Compute the alternative parameters and Vnorm for each cell
        Never called, just for debugging
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



    def compute_q(self):
        indices = self._game.get_indices()

        # compute df_t
        delta_f_t = self._estimator.ucb(indices, delta=1/self.explo_rounds)

        #compute dg_t

        # compute approx derivative of first term using (g(w+eps)-g(w))/eps
        #Warning : may be numerically unstable
        Vw = self.get_Vw(indices)
        VW_eps = self.get_Vw(indices, eps=0.01)
        diff_eps =  self.get_info_ratios(VW_eps) - self.get_info_ratios(Vw)
        diff_eps /= 0.01



        # sq_gaps = (self._means - self._means[self._winner])**2

        delta_g_t = diff_eps + np.sqrt(self._estimator.lls.beta(1/self.explo_rounds) * self._estimator.var(indices))

        return delta_f_t + self.lambda_t * delta_g_t



    def update_alphas(self):
        return 1/np.sqrt(self.p_k), 1/np.sqrt(self.p_k)

    def get_Vw(self, indices, eps=0.):
        X = self._game.get_actions(indices)
        Vw = np.zeros((self._game.d, self._game.d))
        for a in indices:
            Vw = np.add(Vw, ((self.w_t[a]+eps) * np.outer(X[a,:],X[a,:]) ))

        return Vw

    def update_w_l(self, indices):
        X = self._game.get_actions(indices)

        # update w_t


        #compute q_t(x,a)
        q_t= self.compute_q()

        softmax = np.exp(self.alpha_w * q_t)

        self.w_t = np.multiply(self.w_t,softmax)
        self.w_t /= np.sum(self.w_t)

        #update lambda_t :
        if self.reset==True:
            Vw = self.get_Vw(indices)
        else:
            Vw = self.get_Vw(indices, eps=0.001)
        min_Vw_norm = self.get_info_ratios(Vw)

        g_t = min_Vw_norm + np.dot(self.w_t, np.sqrt(self._estimator.lls.beta(1/self.explo_rounds) * self._estimator.var(indices))) - 1/self.z_k
        # print(g_t)
        self.lambda_t = np.max([0,np.min([self.lambda_t - self.alpha_l*g_t, self.lambda_max])])




    def get_info_ratios(self,  V_matrix):
        # Warning, also updates the winner and means
        #

        ## Johannes's implementation
        # I don't understand why the second best gap is also the second best of the ratio
        gaps = self._means[self._winner] - self._means
        gaps[self._winner] = np.inf
        second_best = np.argmin(gaps)
        x_best = self._game.get_actions(self._winner)
        x_2best = self._game.get_actions(second_best)
        # alt_ms = gaps[second_best]**2/self._estimator.lls.var(x_best - x_2best)

        sq_gap = gaps[second_best]**2
        # same as in estimator.lls.var(x) but with the general V_matrix
        cholesky = cho_factor(V_matrix)
        sol_x = cho_solve(cholesky, (x_best-x_2best).T).T
        sq_norm =  np.sum((x_best-x_2best)*sol_x, axis=-1)

        return  sq_gap / sq_norm




    def get_next_action(self):
        indices = self._game.get_indices()


        self._means = self._estimator.lls.mean(self._game.get_actions(indices))
        self._winner = np.argmax(self._means)

        #same as in ids, but article takes delta=1/n and uses n instead of logdet(V_t)
        _t = max(2, self._t)
        beta_t = self._estimator.lls.beta(1/_t ) #* np.log(_t)

        # theta_min, min_V_norm = self.compute_alt(indices, self._estimator.lls.V)
        #
        # self._min_V_norm = min_V_norm

        # compute the minimum "information ratio" as in Eq 80 of Ap K
        self._min_ratio = self.get_info_ratios(self._estimator.lls.V)
        # print(inf_ratios)

        #temporary:
        # nu, V_norm = self.compute_nu(indices)

        # # compute minimum V-norm without winner
        # V_norm[self._winner] = np.Infinity
        # ms = np.min(V_norm)
        # print('ms:'+str(ms))


        # print('min_ratio'+str(self._min_ratio)) # should be equal to self.ms in asymp-ids
        # print(beta_t)

        if self._min_ratio > beta_t:
            #exploitation step
            #recompute at every round because estimator changes
            # print('exploitation !')

            return self._winner

        else:
            chosen_action = np.random.choice(indices, p=self.w_t)
            # updates
            self.explo_rounds += 1
            self.phase_length += 1
            self.update_w_l(indices)

            if self.phase_length == self.p_k:
                #update phase counters
                # print('increasing phase: '+str(self.phase_length))
                self.phase += 1
                self.phase_length = 0
                self.z_k *= np.exp(self.phase / (self.phase -1 ))
                self.z_k = np.floor(self.z_k)
                self.p_k = np.int(self.z_k * np.exp(self.phase)) #removed the 2* in exp
                # print('next phase is ' + str(self.p_k))
                # updates alphas
                self.alpha_w , self.alpha_l = self.update_alphas()
                #reset
                if self.reset:
                    self.w_t =  np.ones(self.K) / self.K #uniform over actions #uniform over actions
                    self.lambda_t = self.lambda_1
            return chosen_action

    def add_observations(self, indices, y):

        super().add_observations(indices, y)

        self._t += 1  # increase step counter each time we get the data
