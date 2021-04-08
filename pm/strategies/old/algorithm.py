from abc import abstractmethod

import numpy as np

# from controller.data_point import DataPoint
from pm.strategies.old.utils import sherrman_morrision_update


class Algorithm:

    def __init__(self, name=None, delta=0.01):
        self._delta = delta
        self.t = 1

        if name is None:
            self._name = type(self).__name__
        else:
            self._name = name

    def initialize(self, domain):
        self._domain = domain
        self._estimator = self.estimator_class(self._domain.d, self._delta)

    @property
    def name(self):
        return self._name

    @property
    def estimator(self):
        return self._estimator

    def reset(self):
        self.estimator.reset()
        self.t = 1

    def get_next_evaluation_point(self):
        raise NotImplementedError

    def add_data_point(self, *data_point):
        self.estimator.add_data_point(*data_point)
        self.t += 1

    def _max_ucb_value(self):
        best_score = -1000000

        for (x,rho) in self._domain:
            score = self.estimator.ucb(x)
            best_score = max(score, best_score)

        return best_score

    def _get_ucb_action(self):
        best_score = -1000000
        best_action = None

        for (x, rho) in self._domain:
            score = self.estimator.ucb(x)
            if score > best_score:
                best_score = score
                best_action = x

        return best_action

    def _get_best_action(self, theta):
        best_score = -1000000
        best_action = None

        for (x, rho) in self._domain:
            score = np.asscalar(theta.reshape(1, -1).dot(x))
            if score > best_score:
                best_score = score
                best_action = x

        return best_action

    def _get_plausible_actions(self):
        """ Returns list plausible actions (x only). """
        max_lcb = self._max_lcb_value()
        plausible_actions = []
        for (x, rho) in self._domain:
            if self.estimator.ucb(x) >= max_lcb:
                plausible_actions.append(x)
        return plausible_actions

    def _max_lcb_value(self):
        best_score = -1000000

        for (x,rho) in self._domain:
            score = self.estimator.lcb(x)
            best_score = max(score, best_score)

        return best_score


    #TODO: Check what is needed from the code below

    def weighted_norm(self, x, W):
        return np.sqrt(np.asscalar(x.reshape(1, -1).dot(W).dot(x.reshape(-1, 1))))

    def weighted_inv_norm(self, x, W):
        return np.sqrt(np.asscalar(x.reshape(1, -1).dot(np.linalg.solve(W, x))))

    def weighted_inv_inc_norm(self, x, W_inv, w_inc):
        W = sherrman_morrision_update(W_inv, w_inc)
        return self.weighted_norm(x, W)

    # def get_best_arm(self, theta):
    #     score = 0
    #     best_arm = None
    #     best_score = -10000
    #     for arm in self.bandit.available_arms:
    #         score = theta.reshape(1,-1).dot(arm)
    #         if score > best_score:
    #             best_score = score
    #             best_arm = arm
    #
    #     return best_arm



    def get_plausible_best_arms(self):
        lcb = -10000
        beta_t = np.sqrt(np.log(np.linalg.det(self.wV)) - 2* np.log(self.delta)) + 1
        for arm in self.bandit.available_arms:
            lcb = max(lcb, self.post_wmean(arm) - beta_t*np.sqrt(self.w_var(arm)))

        plausible_arms = []
      #  print(self.theta)
        for arm in self.bandit.available_arms:
           # print(arm, self.post_mean(arm))
            ucb = self.post_wmean(arm) +  beta_t*np.sqrt(self.w_var(arm))
            if ucb >= lcb:
                plausible_arms.append(arm)

        return plausible_arms
    #
    # def get_fake_plausible_best_arms(self):
    #     lcb = -10000
    #     beta_t = np.sqrt(np.log(np.linalg.det(self.wV)) - 2* np.log(self.delta)) + 1
    #     for arm in self.bandit.available_arms:
    #         lcb = max(lcb, self.post_wmean(arm) - beta_t*(self.w_var(arm)))
    #
    #     plausible_arms = []
    #   #  print(self.theta)
    #     for arm in self.bandit.available_arms:
    #        # print(arm, self.post_mean(arm))
    #         ucb = self.post_wmean(arm) +  beta_t*(self.w_var(arm))
    #         if ucb >= lcb:
    #             plausible_arms.append(arm)
    #
    #     return plausible_arms

    #
    # def beta_t_uniform(self):
    #     return self.beta_t() * self.bandit.noise_bound()


class UniformNoiseBoundMixin():
    def __init__(self, name=None, delta=0.01, estimator_noise_bound=None, environment_noise=None):
        super(UniformNoiseBoundMixin, self).__init__(name=name, delta=delta)
        self._estimator_noise_bound = estimator_noise_bound
        self._environment_noise = environment_noise

    def add_data_point(self, data_point):
        """ if estimator noise bound is set, use this noise bound on the estimator """
        if self._estimator_noise_bound is None:
            super(UniformNoiseBoundMixin, self).add_data_point(data_point)
        else:
            # NOTE: this line lead to an unintentional bug in the first version of the paper. Since for the homoscedastic algorith,s the noise bound is completely ignored, we need to emulate the heteroscedasitc/weighted lls here by adding reweighted data.
            self.estimator.add_data_point(DataPoint(data_point.t, data_point.x/self._estimator_noise_bound, self._estimator_noise_bound, data_point.y/self._estimator_noise_bound))
            self.t += 1

    def get_next_evaluation_point(self):
        """ if environment_noise is set, always get observations at this level """
        (x,rho) = super(UniformNoiseBoundMixin, self).get_next_evaluation_point()

        if not self._environment_noise is None:
            rho = self._environment_noise

        return (x, rho)


class AcquisitonAlgorithm(Algorithm):

    def acquisition_score(self, x, rho):
        raise NotImplementedError

    def acquisition_init(self):
        pass

    def get_next_evaluation_point(self):
        self.acquisition_init()

        best = None
        best_score = -10000000

        for (x, rho) in self._domain:
            score = self.acquisition_score(x, rho)
            if score > best_score:
                best = (x, rho)
                best_score = score
        #print(-best_score, "IDS-D")
        return best
