import numpy as np
from scipy.linalg import cho_solve, cho_factor


class GPRegression:

    def __init__(self, x_train, y_train, kernel, reg=1.):
        assert x_train.shape[0] == y_train.shape[0]
        self.num_train = x_train.shape[0]
        self.x_train = x_train
        self.y_train = y_train
        self.kernel = kernel
        self.reg = reg
        self.K = self.kernel(x_train, x_train) + self.reg * np.eye(self.num_train)
        self.chol_K = cho_factor(self.K)

    def predict(self, x_target):
        Ktx = self.kernel(x_target, self.x_train)
        Ktx_inv = cho_solve(self.chol_K, Ktx.T)
        mean = self.y_train @ Ktx_inv
        var = np.diag(self.kernel(x_target)) - np.sum(Ktx * Ktx_inv.T, axis=1)
        return mean, var

    def beta(self, delta):
        logdet = 2 * np.sum(np.log(np.diag(self.chol_K[0])))
        logdet_reg = logdet + self.num_train*np.log(1/self.reg)
        return (np.sqrt(logdet_reg + 2*np.log(1/delta)) + np.sqrt(self.reg))**2


class DuelingGPRegression:

    def __init__(self, x1_train, x2_train, d_train, kernel, reg=1.):
        assert x1_train.shape[0] == d_train.shape[0]
        assert x2_train.shape[0] == d_train.shape[0]

        self.num_train = x1_train.shape[0]
        self.x1_train = x1_train
        self.x2_train = x2_train
        self.d_train = d_train
        self.kernel = kernel
        self.reg = reg
        self.K = self.kernel(x1_train, x1_train) \
                 + self.kernel(x2_train, x2_train) \
                 - self.kernel(x1_train, x2_train) \
                 - self.kernel(x2_train, x1_train) \
                 + self.reg * np.eye(self.num_train)
        self.chol_K = cho_factor(self.K)

    def precompute_target(self, xtarget):
        self._Ktx = self.kernel(xtarget, self.x1_train) - self.kernel(xtarget, self.x2_train)
        self._Ktx_inv = cho_solve(self.chol_K, self._Ktx.T)
        self._xtarget = xtarget

    def mean(self):
        return self.d_train @ self._Ktx_inv

    def var(self, x):
        x = np.atleast_2d(x)
        Ktx = self.kernel(x, self.x1_train) - self.kernel(x, self.x2_train)
        Ktx_inv = cho_solve(self.chol_K, Ktx.T)
        return np.diag(self.kernel(x)) - np.sum(Ktx * Ktx_inv.T, axis=1)

    def psi(self, x_ref, x_target):
        x_ref = np.atleast_2d(x_ref)
        var_xref = self.var(x_ref)
        var_xtarget = np.diag(self.kernel(x_target)) - np.sum(self._Ktx * self._Ktx_inv.T, axis=1)

        Kt_ref = self.kernel(x_ref, self.x1_train) - self.kernel(x_ref, self.x2_train)
        covar = self.kernel(x_ref, x_target) - np.sum(Kt_ref * self._Ktx_inv.T, axis=1)
        return var_xref + var_xtarget - 2 * covar

    def beta(self, delta):
        logdet = 2 * np.sum(np.log(np.diag(self.chol_K[0])))
        logdet_reg = logdet + self.num_train*np.log(1/self.reg)
        return (np.sqrt(logdet_reg + 2*np.log(1/delta)) + np.sqrt(self.reg))**2