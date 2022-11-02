from scipy.linalg import solve_triangular

from pm.strategies.lls import RegularizedLeastSquares
from pm.strategy import Strategy
import numpy as np

from pm.utils import psd_norm_squared


class E2D(Strategy):
    def __init__(self, game, lls : RegularizedLeastSquares, gamma_power=0.5):
        super().__init__(game)
        self.lls = lls
        self.K = game.k
        self.d = game.d
        self.mu = np.ones(self.K) / self.K
        self.t = 1
        self.I = np.eye(self.K)
        self.gamma_power = gamma_power

    def get_action(self):
        self.update_mu(fw_iter=100)
        return np.random.choice(self.K, p=self.mu)

    def get_weighted(self, mu):
        """
        Compute average feature and covariance matrix under sampling distribution  `mu`
        """
        phi_mu = np.zeros(self.game.d)
        V_mu = np.array([Ma.T @ Ma for Ma in self.game.M]).T @ mu
        return V_mu, phi_mu

    def get_theta_alt(self, theta_hat, V_mu, phi_mu, gamma):
        """
        Compute alternative parameters for each possible maximizer.
        return shape (k, d)
        """
        theta_alt = 1 / (2 * gamma) * np.linalg.solve(V_mu, (self.game.X - phi_mu).T).T + theta_hat

        gaps = np.sum(theta_alt * (self.game.X - phi_mu), axis=1)
        KL = psd_norm_squared(theta_alt - theta_hat, V_mu)
        dec = gaps - gamma * KL
        return dec, theta_alt

    # def g(self, V_mu, phi_mu, theta, gamma):
    #     """
    #     computes the value of the dec for any possible maximizing action.
    #     return shape (k, d)
    #     """
    #     diff_X = self.game.X - phi_mu
    #     gaps = theta @ diff_X.T
    #     bonus = 1 / (4 * gamma) * (psd_norm_squared(diff_X, np.linalg.inv(V_mu)))
    #     return gaps + bonus

    def G(self, theta_alt, theta_hat, b, gamma):
        # print(theta_0)
        gap = - self.game.X @ theta_alt + np.inner(theta_alt, self.game.X[b])
        KL = np.sum( (self.game.M @ (theta_alt - theta_hat)) ** 2, axis=1)
        return gap - gamma * KL

    def update_mu(self, fw_iter=1000):
        gamma = self.t**self.gamma_power
        for i in range(1, fw_iter):
            theta_hat = self.lls.theta
            V_mu, phi_mu = self.get_weighted(self.mu)

            # compute action that maximes the dec
            # values = self.g(V_mu, phi_mu, theta_hat, gamma)
            # b = np.argmax(values)

            # compute alternative parameters for each possible maximizing action
            dec, theta_alt = self.get_theta_alt(theta_hat, V_mu, phi_mu, gamma)
            b = np.argmax(dec)

            # compute gradient of dec w.r.t. sampling distribution
            grad_dec = self.G(theta_alt[b], theta_hat, b, gamma)
            a = np.argmin(grad_dec)

            # frank wolfe step
            lrate = 1 / (self.t + i + 2)
            self.mu = (1 - lrate) * self.mu + lrate * self.I[a]

    def add_observations(self, actions, obs):
        m = self.game.M[actions]
        self.lls.add_data(m, obs)
        self.t += 1