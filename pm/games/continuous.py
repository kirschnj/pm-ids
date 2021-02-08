import numpy as np


def get_grid(*bounds, points_per_dim=50):
    """ helper function to discretize a box domain """
    points = [np.linspace(*lim, points_per_dim) for lim in bounds]
    return np.vstack([x.flatten() for x in np.meshgrid(*points)]).T


class ContinuousGame:

    def __init__(self, bounds, points_per_dim=None, name=None):
        self.d = len(bounds)
        self.bounds = bounds
        self.name = name

        if points_per_dim is not None:
            self.discrete_x = get_grid(*self.bounds, points_per_dim=points_per_dim)
        else:
            self.discrete_x = None

    def __str__(self):
        return self.name


class ContinuousInstance:
    def __init__(self, game, function, noise, confounder):
        self.game = game
        self._function = function
        self._noise = noise
        self._confounder = confounder

    def get_reward(self, x):
        return self._function(x)

    def get_observation(self, x):
        # bandit feedback for now
        return self._function(x)

    def max_reward(self):
        return self._function.best_y

    def get_noisy_observation(self, x):
        """
        shape = (len(indices),m)
        """
        y_exact = self.get_observation(x)
        y = y_exact + self._noise(y_exact.shape)
        if self._confounder is not None:
            y += self._confounder()
            self._confounder.record_outcome(y_exact)
        return y

    def __str__(self):
        return type(self._function).__name__
