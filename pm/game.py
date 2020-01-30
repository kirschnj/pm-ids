import numpy as np

class Game:
    """
    Defines a partial monitoring game.

    - index set
    - action set
    - observation map set
    """

    def get_d(self):
        """
        feature dimension
        """
        return self.get_actions(self.get_indices()[0:1]).shape[1]

    def get_m(self):
        """
        observation dimension
        """
        return self.get_observation_maps(self.get_indices()[0:1]).shape[1]

    def get_indices(self):
        """
        returns list where each element defines an action/observation map
        """
        raise NotImplemented

    def get_actions(self, indices):
        """
        returns list of actions, shape: (len(indices),d)
        """
        raise NotImplemented

    def get_observation_maps(self, indices):
        """
        returns list of observation maps. shape (len(indices),m,d)
        """
        raise NotImplemented

    def id(self):
        """
        identifier used in the directory structure to store the results
        """
        raise NotImplemented


class GameInstance:

    def __init__(self, game, theta, noise, id=""):
        """

        :param game:
        :param noise: function that takes a shape argument
        """
        self._game = game
        self._theta = theta
        self._noise = noise
        self._id = id
        self._max_reward = np.max(self.get_reward(self._game.get_indices()))

    def get_reward(self, indices):
        """
        shape = (len(indices))
        """
        X = self._game.get_actions(indices)
        return X.dot(self._theta)

    def get_max_reward(self):
        """
        maximum reward of the game
        """
        return self._max_reward

    def get_observation(self, indices):
        """
        shape = (len(indices),m)
        """
        AX = self._game.get_observation_maps(indices)
        return AX.dot(self._theta)

    def get_noisy_observation(self, indices):
        """
        shape = (len(indices),m)
        """
        y = self.get_observation(indices)
        return y + self._noise(y.shape)

    def id(self):
        """
        identifier used in the directory structure to store the results
        """
        return self._id