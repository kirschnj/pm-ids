from pm.strategies.gp import GPRegression
from pm.strategies.lls import RegularizedLeastSquares
from pm.strategy import Strategy
import numpy as np
from GPy.kern import RBF


class UCB(Strategy):
    def __init__(self, game, lls : RegularizedLeastSquares, force_homoscedastic = False):
        super().__init__(game)
        self.lls = lls
        self._force_homoscedastic = force_homoscedastic

    def get_action(self):
        ucb = self.lls.ucb(self.game.get_actions())
        return np.argmax(ucb)

    def add_observations(self, actions, obs):
        m = self.game.M[actions]
        if self._force_homoscedastic:
            x = self.game.X[actions]
            rho = np.linalg.norm(x)/np.linalg.norm(m)
            m = x.reshape(len(obs),-1)
            obs = obs * rho

        self.lls.add_data(m, obs)


class GPUCB(Strategy):
    def __init__(self, game, delta, reg=1., beta=None, lengthscale=1.):
        super().__init__(game)
        self._xtrain = np.empty((0, game.d))
        self._ytrain = np.empty(0)
        self.t = 0
        self._rbf = RBF(game.d, lengthscale=lengthscale)
        self.delta = delta
        self.reg = reg
        self._beta = beta

        if game.discrete_x is None:
            raise RuntimeError()
        self.X = game.discrete_x

    def kernel(self, x, y=None):
        return self._rbf.K(x, y)

    def get_action(self):
        if self.t == 0:
            i = np.random.choice(len(self.X))
            return self.X[i]

        gp = GPRegression(self._xtrain, self._ytrain, kernel=self.kernel, reg=self.reg)
        mean, var = gp.predict(self.X)
        if self._beta is None:
            beta = gp.beta(self.delta)
        else:
            beta = self._beta
        ucb = mean + np.sqrt(beta*var)
        return self.X[np.argmax(ucb)]

    def add_observations(self, x, y):
        self._xtrain = np.vstack([self._xtrain, x])
        self._ytrain = np.concatenate([self._ytrain, y.flatten()])
        self.t += 1
