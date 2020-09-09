from datetime import datetime
import sys
import numpy as np
import contextlib
import cvxpy as cp

def timestamp():
    """
    compute timestamp used for filenames
    """
    now = datetime.now()
    return now.strftime("%Y%m%d-%H%M%S.%f")


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.
    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).
    The "answer" return value is one of "yes" or "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default == None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " \
                             "(or 'y' or 'n').\n")

def difference_matrix(X):
    """
    Takes an (l,d) matrix X and returns a (l,l,d) matrix D, s.t.
    D[i,j] = X[i] - X[j]
    """
    l, d = X.shape
    # compute all differences in a (l,l,d) array
    return (np.repeat(X, l, axis=0) - np.tile(X, (l, 1))).reshape(l, l, d)

def compute_nu(estimator, game):
    """
    Compute the alternative parameter for each cell but that of the empirical best arm.
    """
    indices = game._I
    d = game._d
    X = game.get_actions(indices)
    theta = estimator._theta
    V = estimator._V
    nu = np.zeros((len(indices),d))
    C = difference_matrix(X)
    # for each action, solve the quadratic program to find the alternative
    for i in indices:
        x = cp.Variable(game._d)
        q = -2 * (V @ theta)
        G = -C[i,:,:]

        prob = cp.Problem(cp.Minimize(cp.quad_form(x, V) + q.T @ x), [G @  x <= 0])
        prob.solve()

        nu[i:] = x.value

    # check corner cases in the bandit case : can the projected nu have a very large norm ? => regularization ?
    #normalize as per our unit ball hypothesis => creates bugs when the projection on the cone is too close to origin. Also does it make sense ?
    # nu /= np.linalg.norm(nu, axis=1)[:, None]
    return nu

# def optimization(A, Delta, ft):
#
#     d, K, M = A.shape
#     c = 10000
#     X = np.zeros([d, K * M])
#     Delta_new = np.zeros(K*M)
#
#     for m in range(M):
#         Delta_new[m*K : (m+1)*K] = Delta[:, m]
#         for j in range(K):
#            X[:,j + m*K] = A[:, j, m]
#     zero_index = np.where(Delta_new == 0)[0]
#     b = (Delta_new ** 2/ft) # constraint upper bounds
#     b[zero_index] += 0.001
#     a = Delta_new # objective coefficients
#
#     # Construct the optimization problem.
#     T = cp.Variable(K*M)
#     B = cp.Variable((d, d))
#     objective = cp.Minimize(a @ T)
#     constraints = [B == X @ cp.diag(T) @ X.T] + \
#                   [T >= 0] + \
#                   [T <= c] + \
#                   [cp.matrix_frac(x, B) <= bi for x, bi in zip(X.T, b)]
#     problem = cp.Problem(objective, constraints)
#
#     # Report solution.
#     #problem.solve()
#     sol = problem.solve()
#     while sol == None:
#         sol = problem.solve()
#
#     # return solutions
#     return T.value

def lower_bound(game, instance, print_sol=False):
    c = 10000
    ind = game.get_indices()
    A = game.get_actions(ind) #all actions shape (K * d)
    K, d = A.shape
    means = instance.get_reward(ind)
    max_reward = instance.get_max_reward()
    Delta = max_reward - means
    zero_index = np.where(Delta == 0)[0]
    b=(Delta**2 /2)
    b[zero_index] += 0.00001

    a=Delta

    #construct the optimization Problem
    T = cp.Variable(K)
    B = cp.Variable((d,d))
    objective = cp.Minimize(a @ T)
    constraints = [B == A.T @ cp.diag(T) @ A] + \
                  [T >= 0] + \
                  [T <= c] + \
                  [cp.matrix_frac(x,B) <= bi for x,bi in zip(A, b)]


    problem = cp.Problem(objective, constraints)

    #report solution
    sol = problem.solve()
    while sol == None:
        sol = problem.solve()

    if print_sol:
        print('solution vector: '+str(T.value))
    return a @ T.value

@contextlib.contextmanager
def fixed_seed(seed):
    """
    context manager that allows to fixes the numpy random seed
    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
