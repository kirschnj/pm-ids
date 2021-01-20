from pm.game import GameInstance


class Confounded(GameInstance):
    def __init__(self, game, instance : GameInstance, confounding_function=None):
        if confounding_function is None:
            confounding_function = lambda x : 0

        self.confounding_function = confounding_function
        self._original_instance = instance
        self.t = 0
        super().__init__(game, instance._theta, instance._noise)

    def get_reward(self, indices):
        """
        shape = (len(indices))
        """
        return self._original_instance.get_reward(indices) + self.confounding_function(self.t)

    def get_max_reward(self):
        """
        maximum reward of the game
        """
        return self._original_instance.get_max_reward() + self.confounding_function(self.t)

    def get_observation(self, indices):
        """
        shape = (len(indices),m)
        """
        return self._original_instance.get_observation(indices) + self.confounding_function(self.t)

    def get_noisy_observation(self, indices):
        """
        shape = (len(indices),m)
        """
        self.t += 1
        return self._original_instance.get_noisy_observation(indices) + self.confounding_function(self.t)

    def id(self):
        """
        identifier used in the directory structure to store the results
        """
        return "confounded"