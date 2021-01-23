import numpy as np
import logging
import cvxpy as cp

from scipy.linalg import cho_solve, cho_factor

from pm.strategy import Strategy
from pm.utils import difference_matrix, psd_norm_squared

class Solid(Strategy):
    def __init__(self, game, estimator, lambda_1=0., z_0=100, alpha_l=0.1, alpha_w = 0.5, lambda_max=10):
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



    def compute_q(self):
        indices = self._game.get_indices()

        # compute df_t
        delta_f_t = self._estimator.ucb(indices, delta=1/self.explo_rounds)

        #compute dg_t

        # compute approx derivative of first term using (g(w+eps)-g(w))/eps
        #Warning : may be numerically unstable
        Vw = self.get_Vw(indices)
        VW_eps = self.get_Vw(indices, eps=0.01)
        diff_eps = np.min(self.get_info_ratios(VW_eps)) - np.min(self.get_info_ratios(Vw))
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
        Vw = self.get_Vw(indices)
        min_Vw_norm = np.min(self.get_info_ratios(Vw))

        g_t = min_Vw_norm + np.dot(self.w_t, np.sqrt(self._estimator.lls.beta(1/self.explo_rounds) * self._estimator.var(indices))) - 1/self.z_k
        # print(g_t)
        self.lambda_t = np.max([0,np.min([self.lambda_t - self.alpha_l*g_t, self.lambda_max])])


    def get_info_ratios(self,  V_matrix):
        # Warning, also updates the winner and means

        indices = self._game.get_indices()
        X = self._game.get_actions(indices)
        self._means = self._estimator.lls.mean(self._game.get_actions(indices))
        self._winner = np.argmax(self._means)
        X_win = X[self._winner,:]

        sq_norms = psd_norm_squared(X - X_win, V_matrix)
        sq_norms[self._winner] = 1 ## to avoid numerical problems, returns a null ratio
        sq_gaps = (self._means - self._means[self._winner])**2
        sq_gaps[self._winner] = 10000

        return  np.divide(sq_gaps,sq_norms)


    def get_next_action(self):
        indices = self._game.get_indices()
        #same as in ids, but article takes delta=1/n and uses n instead of logdet(V_t)
        beta_t = self._estimator.lls.beta(1/(self._t * np.log(self._t + 2)))  # adding +1 to avoid numerical issues at initialization


        # theta_min, min_V_norm = self.compute_alt(indices, self._estimator.lls.V)
        #
        # self._min_V_norm = min_V_norm

        # compute the minimum "information ratio" as in Eq 80 of Ap K
        inf_ratios = self.get_info_ratios(self._estimator.lls.V)
        # print(inf_ratios)
        self._min_ratio = np.min(inf_ratios)

        # print(self._min_ratio)
        # print(beta_t)

        if self._min_ratio > beta_t:
            #exploitation step
            #recompute at every round because estimator changes
            logging.debug(f"Exploitation round: {self._t}")


            return self._winner

        else:
            chosen_action = np.random.choice(indices, p=self.w_t)
            # updates
            self.explo_rounds += 1
            self.phase_length += 1
            self.update_w_l(indices)

            if self.phase_length == self.p_k:
                #update phase counters
                logging.info('increasing phase: '+str(self.phase_length))
                self.phase += 1
                self.phase_length = 0
                self.z_k *= np.exp(self.phase / (self.phase -1 ))
                self.z_k = np.floor(self.z_k)
                self.p_k = np.int(self.z_k * np.exp(self.phase)) #removed the 2* in exp
                logging.info('next phase is ' + str(self.p_k))
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
