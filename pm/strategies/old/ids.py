from pm.strategies.old.algorithm import Algorithm, AcquisitonAlgorithm
import numpy as np

# from algorithms.estimator import WeightedLeastSquares
from pm.strategies.old.utils import project_onto_simplex, sherrman_morrision_update
from scipy import stats
import math


class Estimator():

    def __init__(self):
        self.reset()

    @property
    def beta(self):
        pass

    def mean(self, x):
        pass

    def std(self, x):
        pass

    def ucb(self, x):
        return self.mean(x) + self.beta * self.std(x)

    def lcb(self, x):
        return self.mean(x) - self.beta * self.std(x)

    def reset(self):
        pass

    def add_data_point(self, data_point):
        pass


class ParameterEstimator(Estimator):

    @property
    def theta(self):
        return self._theta


class LinearEstimator(ParameterEstimator):

    def __init__(self, d, delta):
        self._d = d
        self.delta = delta
        self.reset()
        super(LinearEstimator, self).__init__()

    def reset(self):
        self._V = np.eye(self._d)
        self._Y = np.zeros(self._d).reshape(-1, 1)
        self._theta = np.zeros(self._d).reshape(1, -1)
        self._n = 0
        self._update_cached()

    def add_data_point(self, data):
        """ just updating cache here and counting, adding datapoint needs to be done in child class"""
        self._update_cached()
        self._n += 1

    def _update_cached(self):
        self._theta = np.linalg.solve(self._V, self._Y).reshape(1, -1)
        self._detV = np.linalg.det(self.V)
        self._beta_t = np.sqrt(np.log(self.detV) - 2 * np.log(self.delta)) + 1

    @property
    def V(self):
        return self._V

    def mean(self, x):
        return np.asscalar(self.theta.reshape(1, -1).dot(x))

    @property
    def theta(self):
        return self._theta

    def var(self, x):
        return np.asscalar(x.reshape(1, -1).dot(np.linalg.solve(self.V, x)))

    def std(self, x):
        return np.sqrt(self.var(x))

    def mi(self, x, rho):
        return np.log(1 + self.var(x) / (rho*rho))

    #TODO: Update Iterativly
    @property
    def detV(self):
        return self._detV

    @property
    def beta(self):
        return self._beta_t

    #TODO: Caching/Iterative Update
    @property
    def V_inv(self):
        return np.linalg.inv(self.V)


class RidgeEstimator(LinearEstimator):

    def add_data_point(self, data):
        self._V += data.x.dot(data.x.reshape(1, -1))
        self._Y += data.x * data.y
        super(RidgeEstimator, self).add_data_point(data)



class WeightedLeastSquares(LinearEstimator):

    def add_data_point(self, m, y):
        self._V += np.outer(m, m)
        self._Y += (m*y).reshape(-1, 1)
        # r2 = data.rho * data.rho
        # self._V += data.x.dot(data.x.reshape(1, -1)) / r2
        # self._Y += data.x * data.y / r2
        super(WeightedLeastSquares, self).add_data_point(None)



# this class captures all relevant parameters of the domain
# for now, this is a discrete domain
class Domain():

    def __init__(self):
        self._extended_domain = []

    def set_extended_domain(self, extended_domain):
        self._extended_domain = extended_domain

    def add_point(self, x, rho):
        self._extended_domain.append((x, rho))

    @property
    def extended_domain(self):
        return self._extended_domain

    def set_d(self, d):
        self._d = d

    @property
    def d(self):
        return self._d

    @property
    def size(self):
        return len(self._extended_domain)

    def __iter__(self):
        return iter(self._extended_domain)

class IDS(Algorithm):
    estimator_class = WeightedLeastSquares


    def __init__(self, delta, name=None, skip=1, plausible_actions_only=False):
        super(IDS, self).__init__(delta=delta, name=name)
        self._p = None
        self._skip=skip
        self._plausible_actions_only = plausible_actions_only
        self._total_g = 0
        self._total_sqrt_g = 0

    def reset(self):
        super(IDS, self).reset()

    def get_next_evaluation_point(self):
        if (self.t - 1) % self._skip == 0:


            Delta = np.zeros(self._domain.size)
            g = np.zeros(self._domain.size)
            max_ucb = self._max_ucb_value()

            # if self._plausible_actions_only:
            max_lcb = self._max_lcb_value()


            self._actions = []
            self._init_g()

            i = 0
            for (x,rho) in self._domain:
                if not self._plausible_actions_only or self.estimator.ucb(x) >= max_lcb:
                    Delta[i] =(max_ucb - self.estimator.lcb(x))
                    g[i] = self._get_g(x, rho)
                    self._actions.append((x,rho))
                    i += 1

            self._m = i # number of actions in use
            #print(self._m)
            Delta = Delta[:self._m]  # remove over-allocation
           # g = g[:self._m]  # remove over-allocation
            #g = self._get_g()

            # print(g)

            # return (Delta, g)
            #print(self._m, g, Delta)
            (self._min_i, self._min_j, self._min_alpha) = self.find_minimizer(Delta, g)
            return ((self._min_i, self._min_j, self._min_alpha))
            #print(min_i, min_j, min_alpha)
            #print(min_alpha)
            #print(self.estimator.theta)

            #initialize p
            #p_old = np.copy(p)
            # if self._p is None:
            #     p = np.ones(self._m)/self._m
            # else:
            #     p = self._p
            # g = np.array(g)
            # (p, converged) = self.optimizeGD(p, Delta, g, 1000)
            # # if not converged:
            # #     #try with different starting point
            # #     (p, converged) = self.optimizeGD(np.ones(m)/m, Delta, g, m, 1000)
            # #     if not converged:
            # #         print("restart not converged!")
            #
            # choice = np.random.choice(self._m, p=p)
            # self._p = p
            # return self._actions[choice]

        # sanity check:
        # ee_Delta = 0
        # ee_g = 0
        # S = 1000
        # for j in range(S):
        #     if np.random.binomial(1, self._min_alpha):
        #         ee_Delta += max_ucb - self.estimator.lcb(self._actions[self._min_i][0])
        #         ee_g += self.estimator.mi(self._actions[self._min_i][0],self._actions[self._min_i][1])
        #
        #     else:
        #         ee_Delta += max_ucb - self.estimator.lcb(self._actions[self._min_j][0])
        #         ee_g += self.estimator.mi(self._actions[self._min_j][0],self._actions[self._min_j][1])
        #
        # eepsi = ee_Delta**2/ee_g/S
        # print("Expected Psi: %s" % eepsi, self._min_alpha)
        # print("Expected regret: %s, Expected Regret of UCB action: %s" % (ee_g/S, max_ucb - max_lcb))

        # if self._m < 4:
        #     print("Less then 4 actions left. Current estimator: %s" % self.estimator.theta)
        #     for i in range(self._m):
        #         (x, rho) = self._actions[i]
        #         if i == self._min_i:
        #             print("alpha", self._min_alpha)
        #         if i == self._min_j:
        #             print("alpha", 1- self._min_alpha)
        #         print(self.estimator.mean(x), self.estimator.std(x), self.estimator.beta, self.estimator.ucb(x))

        if np.random.binomial(1, self._min_alpha):
            (x,rho) = self._actions[self._min_i]
        else:
            (x, rho) = self._actions[self._min_j]


        # if self.estimator.ucb(x) == max_ucb:
        #     print("%s, UCB Action Played" % self.t)
        # else:
        #     print("%s, Not" % self.t)
        # self._total_g += self.estimator.mi(x,rho)
        # self._total_sqrt_g += math.sqrt(self.estimator.mi(x,rho))
        return (x, rho)


    def optimizeGD(self, p, Delta, g, T):
        # Gradient Descent
        psi_old = 0
        psi = self.PSI(p, Delta, g)
        i = 0
        eta = np.zeros(self._m)
        while i < T and (i < 100 or (i % 5 or np.abs(psi_old - psi) > 0.00001)):
            psi_old = psi
            pDelta = p.dot(Delta)
            pg = p.dot(g)
            gamma = 0.01
            grad = (2 * pDelta * Delta / pg - np.asscalar(pDelta * pDelta / (pg * pg)) * g)
            p -= gamma * grad

            #eta += grad*grad
            project_onto_simplex(p, self._m)
            # print(p, sum(p))
            # print(self.PSI(self.p, Delta,g))
            psi = self.PSI(p, Delta, g)
            i += 1
        #if(i==500):
         #   print(p)
        #print(i)
        #print(psi, "min value gradient descents")
        return (p, i != T)

    def optimizeEG(self, p, Delta, g, m):
        # Gradient Descent
        psi_old = 0
        psi = self.PSI(p, Delta, g)
        i = 0
        while i < 5000:
            psi_old = psi

            pDelta = p.dot(Delta)
            pg = p.dot(g)
            eta = 0.01* np.sqrt(np.log(m)/500)
            p = p *np.exp( - eta * (2 * pDelta * Delta / pg - pDelta * pDelta / (pg * pg) * g))
            p /= np.sum(p)
            #project_onto_simplex(p, m)
            # print(p, sum(p))
            # print(self.PSI(self.p, Delta,g))
            psi = self.PSI(p, Delta, g)
            i += 1

        print(i, p, sum(p))
        return p

    def find_minimizer(self, Delta, g):
        min_value = 10000000
        min_distr = None
        # min_value_single_point = 100000000
        for i in range(self._m):
            for j in range(i+1):
                (alpha, value) = solve_minimum_ratio(Delta[i], g[i], Delta[j], g[j])

                if value < min_value:
                    min_value = value
                    min_distr = (i, j, alpha)
                # if i == j and min_value_single_point > value:
                #     min_value_single_point = value

        print(min_value, "IDS", self.estimator.beta**2)
        print(alpha*g[i]+(1-alpha)*g[j])

        return min_distr

    def PSI(self, p, Delta, g):
        pDelta = np.asscalar(p.dot(Delta))
        pg = np.asscalar(p.dot(g))

        return pDelta*pDelta/pg

    def ee_g(self, p, g):
        pg = np.asscalar(p.dot(g))
        return pg




class GTheta:
    def _get_g(self, x, rho):
        return self.estimator.mi(x, rho)

    def _init_g(self):
        pass


class GX:
    """ Class which calulates I(x;X). """
    def _init_g(self):
        self._g_X = self._get_X()
        self._g_m = len(self._g_X)
        self._g_Var = [0] * self._g_m
        # precompute variances
        for i in range(self._g_m):
            self._g_Var[i] = self.estimator.var(self._g_X[i])

    def _get_X(self):
        """ get set of actions which is used to calculate g_(x) = I(x;X)
            has to return a list of actions.
        """
        raise NotImplementedError

    def _get_g(self, x, rho):
        """ Calculate average I(x;x') for all x' in X"""
        total_g = 0
        for i in range(self._g_m):
            X_V_x = np.asscalar(self._g_X[i].reshape(1, -1).dot(np.linalg.solve(self.estimator.V, x)))
            total_g += math.log(1/(1- X_V_x**2/self._g_Var[i] * 1/(rho*rho + self.estimator.var(x))))
        return total_g / self._g_m


class GUCB(GX):
    """ Mutual information w.r.t UCB action """
    def _get_X(self):
        return [self._get_ucb_action()]

class GPA(GX):
    """ Mutual information w.r.t to all plausible actions """
    def _get_X(self):
        return self._get_plausible_actions()

class GTS(GX):
    """ Mutual information w.r.t TS action """
    def _get_X(self):
        sampled_theta = np.random.multivariate_normal(self.estimator.theta.reshape(-1), self.estimator.V_inv)
        return [self._get_best_action(sampled_theta)]

class GE(GX):
    """ Mutual information w.r.t multiple actions sampled from posterior distribution """
    def _get_X(self):
        sampled_theta = np.random.multivariate_normal(self.estimator.theta.reshape(-1), self.estimator.V_inv, size=10)
        return [self._get_best_action(theta) for theta in sampled_theta]

class MVMI:
    def _init_g(self):
        self._S = 50
        self._g_max_values = self._get_max_samples(self._estimator.theta, self._estimator.V_inv, self._S)

    # this is way faster than scipy version
    @staticmethod
    def _norm_pdf(x):
        return math.exp(-x ** 2 / 2) * GValue.sqrt_2_pi

    # this is way faster than scipy version
    @staticmethod
    def _norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    def _get_g(self, x, rho):
        var_x = self.estimator.var(x)
        mean_x = self.estimator.mean(x)
        mi_list = []

        for y_max in self._g_max_values:
            mi_list.append(self.approximate_mi(mean_x, var_x, rho*rho, y_max))
        g= np.nanmean(mi_list)
        return g

    def approximate_mi(self,y,var, noise_var, y_max):
        beta = (y_max - y)/np.sqrt(var)
        Z =self._norm_cdf(beta)
        cond_var =  noise_var + var*(1- beta * self._norm_pdf(beta)/Z - (self._norm_pdf(beta)/Z)**2)
        return  (self.entropy_normal(var + noise_var) - self.entropy_normal(cond_var))

    def entropy_normal(self, var):
        return 0.5*np.log(var) # without the multiplicate factors in the log, as they cancel in the mutual information

    def _get_max_samples(self, theta, V_inv, count):
        sampled_theta = np.random.multivariate_normal(theta.reshape(-1), V_inv, size=count)

        max_values = [0] * count
        i = 0
        for theta in sampled_theta:
            max_value = -1000
            for (x, rho) in self._domain:
                max_value = max(max_value, np.asscalar(theta.reshape(1,-1).dot(x)))
            max_values[i] = max_value
            i += 1

        return max_values


class GValue:
    sqrt_2_pi = 1/math.sqrt(2 * math.pi)

    def _init_g(self):
        self._S = 10
        self._g_max_values = self._get_max_samples(self._estimator.theta, self._estimator.V_inv, self._S)

    def _get_g(self, x, rho):

        g_x = 0
        for y in self._g_max_values:
            gamma_star = (y -  self.estimator.mean(x))/self.estimator.std(x)
            cdf_gamma_star = GValue._norm_cdf(gamma_star)
            g_x += (gamma_star * GValue._norm_pdf(gamma_star)/(2 *cdf_gamma_star ) - np.log(cdf_gamma_star))

        # if g_x <= 0:
            # raise Exception("mutual information was 0")
        return g_x/self._S #max(g_x/self._S, 0.01)

    # this is way faster than scipy version
    @staticmethod
    def _norm_pdf(x):
        return math.exp(-x ** 2 / 2) * GValue.sqrt_2_pi

    # this is way faster than scipy version
    @staticmethod
    def _norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    def _get_g2(self):
        S = 10
        max_values = self._get_max_samples(self._estimator.theta, self._estimator.V_inv, S)
        g = [0] * self._m

        i = 0
        for (x,rho) in self._actions:
            g_x = 0
            for y in max_values:
                gamma_star = (y- self.estimator.mean(x))/self.estimator.std(x)
                cdf_gamma_star = stats.norm.cdf(gamma_star)
                g_x += (gamma_star * stats.norm.pdf(gamma_star)/(2 *cdf_gamma_star ) - np.log(cdf_gamma_star))

            g[i] = max(g_x/S, 0.000001)

            i += 1
        return g

    def _get_g_samples(self):
        epsilon = 0.0001
        g = [0] * self._domain.size
        g_ = [0] * self._domain.size

        max_values = self._get_max_samples(self._estimator.theta, self._estimator.V_inv, 1000)
        entropy = self._estimate_entropy(max_values)

        i = 0
        for (x,rho) in self._domain:
            V_inv_x = sherrman_morrision_update(self._estimator.V_inv, x/rho)
            max_values_x = self._get_max_samples(self._estimator.theta, V_inv_x, 100)
            entropy_x = self._estimate_entropy(max_values_x)
            g[i] = max(entropy - entropy_x, 0.00000001)
            #g_[i] = entropy - entropy_x
            i += 1
        #print((np.array(g_) < 0).sum())
        return g

    def _get_max_samples(self, theta, V_inv, count):
        sampled_theta = np.random.multivariate_normal(theta.reshape(-1), V_inv, size=count)

        max_values = [0] * count
        i = 0
        for theta in sampled_theta:
            max_value = -1000
            for (x, rho) in self._domain:
                max_value = max(max_value, np.asscalar(theta.reshape(1,-1).dot(x)))
            max_values[i] = max_value
            i += 1

        return max_values

    def _estimate_entropy(self, samples):
        kernel = stats.gaussian_kde(samples, 0.1)
        kernel_values = kernel.evaluate(samples)
        return np.mean(- np.log(kernel_values)*kernel_values)



class DeterministicIDS(AcquisitonAlgorithm):
    estimator_class = WeightedLeastSquares

    def acquisition_init(self):
        self._max_ucb = self._max_ucb_value()
        self._actions = self._domain._extended_domain
        self._m = self._domain.size
        self._init_g()

    def acquisition_score(self, x, rho):
        regret_surrogate = self._max_ucb - self.estimator.lcb(x)
        g = self._get_g(x,rho)
        ratio = calculate_ratio(regret_surrogate, g)
        return -ratio


class UCBIDS(Algorithm):
    color = "darkgreen"
    linestyle = "-."
    estimator_class = WeightedLeastSquares

    def get_next_evaluation_point(self):
        max_ucb = -10000
        x_ucb = None
        rho_ucb = None
        for (x, rho) in self._domain:
            ucb_value = self._estimator.ucb(x)
            if ucb_value > max_ucb or (ucb_value == max_ucb and rho < rho_ucb):
                max_ucb = ucb_value
                x_ucb = x
                rho_ucb = rho

        ucb_g = self._estimator.mi(x_ucb, rho_ucb)
        Delta_ucb = max_ucb - self._estimator.lcb(x_ucb)

        min_psi = 1000000
        min_second_x = None
        min_second_rho = None
        min_alpha = None
        for (x, rho) in self._domain:
            Delta_2 = ucb_value - self._estimator.lcb(x)
            g_2 = self._estimator.mi(x, rho)
            (alpha, psi) = solve_minimum_ratio(Delta_ucb, ucb_g, Delta_2, g_2)

            if psi < min_psi:
                min_psi = psi
                min_alpha = alpha
                min_second_x = x
                min_second_rho = rho

        return sample_action(x_ucb, rho_ucb, min_second_x, min_second_rho, min_alpha)

    def acquisition_score(self, x, rho):
        ratio = calculate_ratio(self._max_ucb - self.estimator.lcb(x), self.estimator.mi(x, rho))
        return - ratio


def solve_minimum_ratio(Delta_1, g_1, Delta_2, g_2):
    if Delta_1 == Delta_2:
        alpha = g_1 > g_2
    elif Delta_1 == 0:
        alpha = 1
    elif g_1 != g_2:
        alpha = (Delta_2 * (g_1 + g_2) - 2 * Delta_1 * g_2) / ((Delta_1 - Delta_2) * (g_1 - g_2))

        alpha = max(0, min(alpha, 1))
        # if alpha is not in [0,1], take smaller ratio of either 1 or 2
        # if alpha < 0 or alpha > 1:
        #     alpha = Delta_1**2/g_1 < Delta_2**2/g_2

    elif Delta_2 == 0:
        alpha = 0
    elif g_1 == g_2:
        alpha = Delta_1 < Delta_2
        if alpha == 0 or g_1 < 0.0000000001:  # infinite information ratio (no information gain)n
            return (alpha, 100000000)
    else:
        raise Exception("Uncovered Case!")

    if alpha == 0 and g_2 < 0.0000000001 or alpha == 1 and g_1 < 0.0000000001:
        return (alpha, 100000000)


    # value = calculate_ratio(Delta_1, g_1, Delta_2, g_2, alpha)
    # not calling the calculate_ratio, to avoid a few 1000 function calls
    value = (alpha * (Delta_1 - Delta_2) + Delta_2) ** 2 / (alpha * (g_1 - g_2) + g_2)
    return (alpha, value)


def calculate_ratio(Delta_1, g_1, Delta_2=0, g_2=0, alpha=1):
    return (alpha * (Delta_1 -Delta_2) +  Delta_2) ** 2 / (alpha * (g_1 - g_2) +  g_2)


def sample_action(x_1, rho_1, x_2, rho_2, alpha):
    if np.random.binomial(1, alpha):
        return (x_1, rho_1)
    else:
        return (x_2, rho_2)


class IDSValue(IDS, GValue):
    pass

class DIDSValue(DeterministicIDS, GValue):
    pass

class IDSTheta(IDS, GTheta):
    pass

class DIDSTheta(DeterministicIDS, GTheta):
    pass

class IDSUCB(IDS, GUCB):
    pass

class DIDSUCB(DeterministicIDS, GUCB):
    pass

class IDSTS(IDS, GTS):
    pass

class DIDSTS(DeterministicIDS, GTS):
    pass

class IDSPA(IDS, GPA):
    pass

class DIDSPA(DeterministicIDS, GE):
    pass

class IDSE(IDS, GE):
    pass

class DIDSE(DeterministicIDS, GE):
    pass

class IDSMV(IDS, MVMI):
    pass

class DIDSMV(DeterministicIDS, MVMI):
    pass
