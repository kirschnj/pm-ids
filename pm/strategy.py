

class Strategy:
    """
    Base class for a strategy
    """

    def __init__(self, game, estimator):
        self._game = game
        self._estimator = estimator

    def add_observations(self, indices, y):
        self._estimator.add_data(indices, y)

    def get_next_action(self):
        raise NotImplemented

    def __str__(self):
        return type(self).__name__