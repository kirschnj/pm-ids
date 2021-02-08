import numpy as np

class SquaredExponential:
    def __init__(self, d=2, sigma=1., distance=0.5):
        super().__init__()
        if distance > 1.:
            raise RuntimeError()

        self.d = d
        self.distance = distance
        x0 = np.random.normal(size=self.d)
        self.x0 = distance * x0 / np.linalg.norm(x0)
        self.bounds = np.array([-np.ones(self.d), np.ones(self.d)]).T
        self.best_y = 1.
        self.best_x = np.zeros(self.d)
        self.sigma = sigma

    def __call__(self, x):
        x = np.atleast_2d(x)
        x_squared = np.sum((x*x)/self.sigma**2/2, axis=-1)
        return np.exp(-x_squared)

class Camelback:
    """
    Camelback benchmark function.
    """
    def __init__(self):
        super().__init__()
        self.d=2
        # self.x0 = np.array([0.5, 0.2])
        self.x0 = np.array([-0.12977758051079197, 0.2632096107305229])
        self.best_y = 1.03162842
        self.bounds = np.array([[-2,2],[-1,1]])

    def __call__(self, x):
        x = np.atleast_2d(x)
        xx = x[:,0]
        yy = x[:,1]
        y = (4. - 2.1*xx**2 + (xx**4)/3.)*(xx**2) + xx*yy + (-4. + 4*(yy**2))*(yy**2)
        return np.maximum(-y, -2.5)
