import numpy as np
import sys

def sherrman_morrision_update(A_inv, x_inc):
        A_x = A_inv.dot(x_inc)
        return  A_inv - A_x.dot(x_inc.reshape(1,-1)).dot(A_inv)/(1 + np.asscalar(x_inc.reshape(1,-1).dot(A_x)))


# https://arxiv.org/abs/1309.1541
def project_onto_simplex(p, m):
    u = np.sort(np.copy(p))[::-1]
    rho = -1
    s = 0
    s_rho = 0
    j = 1

    for j in range(m):
        s += u[j]
        if u[j] + 1 / (j + 1) * (1 - s) > 0:
            rho = j + 1
            s_rho = s
        else:
            break

    l = 1 / rho * (1 - s_rho)

    for i in range(m):
        p[i] = max(p[i] + l, 0)


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
#
# def inv22(mat):
#     return np.array([mat[1,1],-mat[0,1],-mat[1,0],mat[0,0]]).reshape(2,2)/(mat[0,0]*mat[1,1] - mat[0,1]*mat[1,0])
#
# def l2(x):
#     return np.sqrt(np.sum(x * x))
#
# def l1(x):
#     return np.sum(np.abs(x))
#
# def l05(x):
#     return np.sum(np.sqrt(np.abs(x)))
#
# def lp(x,p):
#     return np.power(np.sum(np.power(np.abs(x), p)), 1/p)


def l2(x):
    return np.sqrt(np.sum(x * x))