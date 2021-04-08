

class Strategy:
    """
    Base class for a strategy
    """

    def __init__(self, game):
        self.game = game

    def add_observations(self, actions, obs):
        raise NotImplemented

    def get_action(self):
        raise NotImplemented

    def __str__(self):
        return type(self).__name__