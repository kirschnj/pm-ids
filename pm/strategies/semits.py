import numpy as np
from pm.strategy import Strategy


class SemiparametricTS(Strategy):

    def __init__(self, game, delta):
        self.game = game
        self.t = 2
        self.delta = delta
        self.B = np.eye(self.game.d)
        self.Xy = np.zeros(self.game.d)


    def get_next_action(self):
        # compute plausible actions
        d = self.game.d
        indices = self.game.get_indices()
        k = self.game.k
        X = self.game.get_actions(indices)

        # robust least squares
        mu = np.linalg.solve(self.B, self.Xy)

        v = np.sqrt(2 * np.log(self.t/self.delta))

        T = 1000
        sample = np.random.multivariate_normal(mu, v**2 * np.linalg.inv(self.B), size=T)
        means = X.dot(sample.T)

        # count statistic of optimal arm
        w = np.zeros(k)
        best_arm_samples = np.argmax(means, axis=0)
        for i in best_arm_samples:
            w[i] += 1
        w /= T

        # 'w' is the TS distribution
        xmean = w.dot(X)
        X_centered = X - xmean
        i_choice = best_arm_samples[0]
        x_choice = X[i_choice]

        self.B += np.outer(x_choice - xmean, x_choice - xmean)
        for w_i, X_c in zip(w, X_centered):
            self.B += w_i*np.outer(X_c, X_c)

        self.m = xmean
        self.x_choice = x_choice
        return i_choice

    def add_observations(self, indices, y):
        self.Xy += 2*(self.x_choice - self.m)*y.flatten()