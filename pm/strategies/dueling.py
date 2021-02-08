
from pm.game import Game
from pm.strategy import Strategy
import numpy as np


class Two(Strategy):

    def __init__(self, base_strategy : Strategy):
        self.base_strategy = base_strategy
        self._data_buffer = []
        self._action = None
        super().__init__(None, None)


    def add_observations(self, indices, y):
        self._data_buffer.append(y)

        if len(self._data_buffer) == 2:
            res = (self._data_buffer[0] - self._data_buffer[1])/2
            # if self.flip
            self.base_strategy.add_observations([self._action], res)
            self._data_buffer = []
            self._action = None

    def get_next_action(self):
        if self._action is None:
            self._action = self.base_strategy.get_next_action()

        # flip actions with probability 1/2
        self.flip = np.random.binomial(1, 0.5)
        if self.flip:
            self._action = (self._action[1], self._action[0])

        return self._action[len(self._data_buffer)]

    def id(self):
        return self.base_strategy.id() + "-two"
