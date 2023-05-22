from scipy.linalg import solve_triangular

from pm.strategies.lls import RegularizedLeastSquares
from pm.strategy import Strategy
import numpy as np

from pm.utils import psd_norm_squared
import matplotlib.pyplot as plt

class E2D(Strategy):
    def __init__(self,
                 game,
                 lls : RegularizedLeastSquares,
                 delta_f=False,
                 anytime_lambda=False,
                 fixed_lambda=0,
                 exploration_multiplier=1,
                 ):
        super().__init__(game)
        self.lls = lls
        self.K = game.k
        self.d = game.d
        self.mu = np.ones(self.K) / self.K
        self.t = 1
        self.I = np.eye(self.K)

        self.delta_f = delta_f

        self.anytime_lambda = anytime_lambda
        self.fixed_lambda = fixed_lambda
        self.exploration_multiplier = exploration_multiplier

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

    def G(self, theta_alt, theta_hat, b, gamma):
        # print(theta_0)
        gap = - self.game.X @ theta_alt + np.inner(theta_alt, self.game.X[b])
        KL = np.sum( (self.game.M @ (theta_alt - theta_hat)) ** 2, axis=1)
        return gap - gamma * KL

    def get_lambda(self):
        if self.fixed_lambda != 0:
            return np.linspace(0, self.fixed_lambda, 2)[1:]
        if not self.anytime_lambda:
            return np.linspace(0, self.exploration_multiplier * self.t **0.5, 2)[1:]
        if self.anytime_lambda:
            return np.linspace(0, self.exploration_multiplier * self.t **0.5, 20)[1:]

    def update_mu(self, fw_iter=1000):
        lambdas = self.get_lambda()

        eps_sq = self.exploration_multiplier * self.d/self.t

        objs = []
        mus = []

        for l in lambdas:

            # middle of the simplex
            mu = self.mu.copy()

            for i in range(1, fw_iter):
                theta_hat = self.lls.theta
                V_mu, phi_mu = self.get_weighted(mu)

                # compute alternative parameters for each possible maximizing action
                dec, theta_alt = self.get_theta_alt(theta_hat, V_mu, phi_mu, l)
                b = np.argmax(dec)

                # compute gradient of dec w.r.t. sampling distribution
                grad_dec = self.G(theta_alt[b], theta_hat, b, l)
                a = np.argmin(grad_dec)

                final_obj = dec[b]
                final_mu = mu

                # frank wolfe step
                lrate = 1 / (self.t + i + 2)
                mu = (1 - lrate) * mu + lrate * self.I[a]

            mus.append(final_mu.copy())
            objs.append(final_obj + l * eps_sq)

        # if self.t % 10 == 1:
        #     print(f"plotting at time {self.t}")
        #     plt.plot(lambdas, objs)
        #     plt.show()

        self.mu = mus[np.argmin(objs)]
        print(f"Iteration: {self.t}, lambda: {lambdas[np.argmin(objs)]}, best_obj: {np.min(objs)}")

    def add_observations(self, actions, obs):
        m = self.game.M[actions]
        self.lls.add_data(m, obs)
        self.t += 1