import numpy as np
from scipy.spatial import ConvexHull


class Game:
    """
    Defines a partial monitoring game.

    - index set
    - action set
    - observation map set
    """
    def __init__(self, X, M, name=None):
        self._d = X.shape[-1]
        self._k = X.shape[-2]
        if X.ndim == 2 and M.ndim == 2:
            M = M.reshape(self._k, 1, self._d)
        self._m = M.shape[-2]
        self._X = X
        self._M = M
        self._name = name

        self._pareto_actions = None

    @property
    def d(self):
        """
        feature dimension
        """
        return self._d

    @property
    def m(self):
        """
        observation dimension
        """
        return self._m

    @property
    def k(self):
        """
        number of actions
        """
        return self._k

    @property
    def X(self):
        return self._X

    @property
    def M(self):
        return self._M

    def get_actions(self):
        """
        returns list of actions, shape: (len(indices),d)
        """
        return self.X


    def get_pareto_actions(self):
        if self._pareto_actions is None:
            hull = ConvexHull(self.get_actions())
            self._pareto_actions = hull.vertices

        return self._pareto_actions

    def get_observation_maps(self):
        """
        returns list of observation maps. shape (len(indices),m,d)
        """
        return self._M

    def get_global_obs_set(self):
        perm = np.random.permutation(self.k)
        V_all = np.zeros((self.d, self.d))
        for m in self.M:
            V_all += m.T @ m
        rank = np.linalg.matrix_rank(V_all)

        V = np.zeros((self.d, self.d))
        obs_set = []
        explore_num = 0
        for i in np.arange(self.k)[perm]:
            M = self.M[i]
            if np.linalg.norm(M) < 1e-20:
                continue

            V += M.T @ M
            obs_set.append(i)
            explore_num += 1
            if np.linalg.matrix_rank(V) >= rank:
                break
        return obs_set, explore_num, V

    def __str__(self):
        if self._name is not None:
            return self._name
        return type(self).__name__

class ContextualGame(Game):

    def __init__(self, X, M, cdistr, name=None):
        super().__init__(X, M, name=name)
        self.cdistr = cdistr
        self.num_context = len(cdistr)
        self.Xavg = np.sum((X.T * cdistr).T, axis=0)

    def get_actions(self, context=None):
        if context is None:
            return super().get_actions()
        return self.X[context]

    def get_observation_maps(self, context=None):
        if context is None:
            return super().get_observation_maps()
        return self.M[context]

    def get_pareto_actions(self, context):
        if context is not None:
            return super().get_pareto_actions()
        hull = ConvexHull(self.X[context])
        return hull.vertices


class GameInstance:
    def __init__(self, game, theta, noise, id=None, confounder=None):
        """
        :param game:
        :param noise: function that takes a shape argument
        """
        self._game = game
        self._theta = theta
        self._noise = noise
        self._confounder = confounder
        self._id = id
        if not game.__class__ is ContextualGame:
            self._max_reward = np.max(self.get_reward())

    def get_reward(self, actions=None, context=None):
        """
        shape = (len(indices))
        """
        X = self._game.get_actions()

        if context is not None:
            X = X[context]

        if actions is not None:
            X = X[actions]

        return X.dot(self._theta)

    def max_reward(self, context=None):
        """
        maximum reward of the game
        """
        if context is None:
            return self._max_reward
        else:
            return np.max(self.get_reward(context))

    def min_gap(self):
        """
        compute minimum gap
        """
        if self._game.__class__ is ContextualGame:
            raise NotImplemented

        rewards = self.get_reward()
        winner = np.argmax(rewards)
        gaps = rewards[winner] - rewards
        gaps[winner] = np.Infinity
        return np.min(gaps)

    def get_observation(self, action, context=None):
        """
        shape = (len(indices),m)
        """
        AX = self._game.get_observation_maps()

        if context is not None:
            AX = AX[context, action]
        else:
            AX = AX[action]
        return AX.dot(self._theta)

    def get_noisy_observation(self, action, context=None):
        """
        shape = (len(indices),m)
        """
        y_exact = self.get_observation(action, context)
        y = y_exact + self._noise(y_exact.shape)
        if self._confounder is not None:
            y += self._confounder()
            self._confounder.record_outcome(y_exact)
        return y

    def __str__(self):
        if self._id is not None:
            return self._id
        return type(self).__name__