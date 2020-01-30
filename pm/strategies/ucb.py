from pm.strategy import Strategy
import numpy as np

class UCB(Strategy):

    def get_next_action(self):
        indices = self._game.get_indices()
        ucb = self._estimator.ucb(indices)
        return indices[np.argmax(ucb)]

    def id(self):
        return "ucb"
