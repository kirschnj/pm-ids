from pm.game import GameInstance, Game
from itertools import product
import numpy as np

def split_indices(indices):
    i0 = [i[0] for i in indices]
    i1 = [i[1] for i in indices]
    return i0, i1


class ConfoundedToDuelingGame(Game):
    """
    Reduction base class for confounded games.
    """
    """
    Two point reduction interface for confounded games.
    """

    def __init__(self, original_game, two_point=False):
        self.original_game = original_game
        self.two_point = two_point
        super().__init__()

    @property
    def d(self):
        return self.original_game.d

    @property
    def discrete_x(self):
        return self.original_game.discrete_x

    def get_m(self):
        return self.original_game.get_m()

    def get_base_indices(self):
        return self.original_game.get_indices()

    def get_indices(self):
        return [*product(self.original_game.get_indices(), self.original_game.get_indices())]

    def get_actions(self, indices):
        """
        returns list of actions, shape: (len(indices),d)
        """
        i0, i1 = split_indices(indices)
        factor = 1. if self.two_point else 0.5
        return factor * (self.original_game.get_actions(i0) + self.original_game.get_actions(i1))

    def get_base_actions(self, indices):
        return self.original_game.get_actions(indices)

    def get_observation_maps(self, indices):
        """
        returns list of observation maps. shape (len(indices),m,d)
        """
        i0, i1 = split_indices(indices)
        return self.original_game.get_observation_maps(i0) - self.original_game.get_observation_maps(i1)


class ConfoundedToDuelingInstance(GameInstance):

    def __init__(self, original_instance, two_point=False, compensate=False):
        self._original_instance = original_instance
        self.two_point = two_point
        self.compensate = compensate
        self.last_obs = 0.

    def get_reward(self, indices):
        i0, i1 = split_indices(indices)
        factor = 1. if self.two_point else 0.5
        return factor*(self._original_instance.get_reward(i0) + self._original_instance.get_reward(i1))

    def max_reward(self):
        """
        maximum reward of the game
        """
        factor = 2. if self.two_point else 1.
        return factor * self._original_instance.max_reward()

    def min_gap(self):
        """
        compute minimum gap
        """
        return self._original_instance.min_gap()

    def get_noisy_observation(self, indices):
        """
        shape = (len(indices),m)
        """
        i0, i1 = split_indices(indices)
        B = np.random.binomial(1, 0.5)
        if B:
            i0, i1 = i1, i0

        if self.two_point:
            # two point reduction method
            y0 = self._get_noisy_observation(i0)
            y1 = self._get_noisy_observation(i1)
            return (-1)**B * (y0 - y1)
        else:
            # one point reduction method
            return 2 * (-1)**B * self._get_noisy_observation(i0)

    def _get_noisy_observation(self, i):
        # return self._original_instance.get_noisy_observation(i)
        last_obs = self.last_obs
        self.last_obs = self._original_instance.get_noisy_observation(i)
        if self.compensate:
            return self.last_obs - last_obs
        return self.last_obs
