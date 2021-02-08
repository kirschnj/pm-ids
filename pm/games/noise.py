import numpy as np


def noise_normal(var=1):
    return lambda size : _noise_normal(size, var=var)

def _noise_normal(size, var=1.):
    """
    helper function that generates Gaussian observation noise
    """
    return np.random.normal(0, scale=var**(1/2), size=size)


class Confounder:
    def __init__(self):
        self.t = 0

    def record_outcome(self, y):
        self.t += 1

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class PeriodicDrift(Confounder):

    def __call__(self, *args, **kwargs):
        return np.sin(0.2*self.t) - 0.1*self.t


class NegativeDrift(Confounder):

    def __call__(self, *args, **kwargs):
        return -0.1*self.t


class NegativeRepeat(Confounder):
    def __init__(self):
        super().__init__()
        self.c = 0

    def __call__(self, *args, **kwargs):
        return self.c

    def record_outcome(self, y):
        super().record_outcome(y)
        self.c = -y

class AutoCalibration(Confounder):
    def __init__(self):
        super().__init__()
        self._last_obs = []
        self._last_mean = .0
        self._max_obs = 5
        self._threshold = 0.2

    def __call__(self, *args, **kwargs):
        return - self._last_mean

    def record_outcome(self, y):
        super().record_outcome(y)
        # print(y)
        # print(self._last_obs)

        # list full pop
        if len(self._last_obs) == self._max_obs:
            self._last_obs.pop(0)
        self._last_obs.append(y.item())

        # print(np.mean(self._last_obs))
        if len(self._last_obs) == self._max_obs:
            if np.abs(np.mean(self._last_obs) - self._last_mean) > self._threshold:
                # print(f"recalibrate at {self.t}")
                self._last_mean = np.mean(self._last_obs) - self._threshold / 2


class PositiveRepeat(Confounder):
    def __init__(self):
        super().__init__()
        self.c = 0

    def __call__(self, *args, **kwargs):
        return self.c

    def record_outcome(self, y):
        super().record_outcome(y)
        self.c = y


class NegativeRepeatTwo(Confounder):
    def __init__(self):
        super().__init__()
        self.c1 = 0
        self.c2 = 0

    def __call__(self, *args, **kwargs):
        return self.c2

    def record_outcome(self, y):
        super().record_outcome(y)
        self.c2 = self.c1
        self.c1 = -y


class MinusBest(Confounder):
    def __init__(self):
        super().__init__()
        self.best_seen = 0.
        self.c = 0.

    def __call__(self, *args, **kwargs):
        return -self.c

    def record_outcome(self, y):
        super().record_outcome(y)
        self.best_seen = max(y, self.best_seen)
        if np.abs(y - self.best_seen) <= 1e-10:
            self.c = -1.
        else:
            self.c = 0


class AlternatingMinusBest(Confounder):
    def __init__(self):
        super().__init__()
        self.best_seen = 0.
        self.c = 0.

    def __call__(self, *args, **kwargs):
        return -self.c

    def record_outcome(self, y):
        super().record_outcome(y)
        self.best_seen = max(y, self.best_seen)
        if np.abs(y - self.best_seen) <= 1e-10 and self.c == 0.:
            self.c = -1.
        else:
            self.c = 0


class Bernoulli(Confounder):

    def __call__(self, *args, **kwargs):
        return np.random.binomial(1, 0.5)


class NegativeBernoulli(Confounder):

    def __call__(self, *args, **kwargs):
        return -np.random.binomial(1, 0.5)


class PhasedOffset(Confounder):
    def __init__(self):
        super().__init__()
        self.best_seen = 0.
        self.c = 0.

    def __call__(self, *args, **kwargs):
        return self.c

    def record_outcome(self, y):
        super().record_outcome(y)
        if self.t % 50 == 0:
            self.c = np.random.uniform(-1, 1)



