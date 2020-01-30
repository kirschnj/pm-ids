from datetime import datetime
import sys
import numpy as np
import contextlib

def timestamp():
    # Converting datetime object to string
    dateTimeObj = datetime.now()
    return dateTimeObj.strftime("%Y%m%d-%H%M%S.%f")


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