import argparse
import numpy as np
import os
import json
import uuid
import shutil

from pm import aggregate
from pm.estimator import RegularizedLeastSquares, RegretEstimator
from pm.game import GameInstance
from pm.games.bandit import Bandit
from pm.games.pm import GenericPM
from pm.strategies.ids import IDS, full, directed2, directeducb, directed3, info_game
from pm.strategies.ucb import UCB
from pm.utils import query_yes_no, timestamp, fixed_seed, compute_nu, lower_bound


def noise_normal(size):
    """
    helper function that generates Gaussian observation noise
    """
    return np.random.normal(0, 1, size=size)


def rbf(x, y):
    """
    shape (n,d)
    """
    return np.exp(- np.sum((x - y) ** 2, axis=1))


def phi(x, X0):
    """
    rbf features for point x and domain X0
    """
    return rbf(x, X0)


def simple_bandit(**params):
    """
    game factory for simple bandit game
    """
    with fixed_seed(params.get('seed', None)):
        X = np.random.normal(size=12).reshape(6, 2)
        X = X / np.linalg.norm(X, axis=1)[:, None]

    game = Bandit(X, id="simple_bandit")
    instance = GameInstance(game, theta=np.array([1., 0.]), noise=noise_normal)

    return game, instance

def counter_example(**params):
    """
    game factory for the counter-example in the End of Optimism
    """
    # alpha = 0.25 such that 8\alpha\epsilon =2\epsilon as in Figure 1
    eps = 0.01
    X = np.array([[1.,0.],[1-eps,2*eps],[0.,1.]])

    game = Bandit(X, id="counter_example")
    instance = GameInstance(game, theta=np.array([1.,0.]), noise=noise_normal)

    return game, instance


def laser(**params):
    """
    game factory for laser experiment
    """
    indirect = params['laser_indirect']

    grid = np.arange(-1, 1.1, 0.5)  # 1.1 to include 1
    d = len(grid) ** 2
    X0 = np.array(np.meshgrid(grid, grid)).T.reshape(d, 2)


    # these are parameter settings that can be played
    grid2 = np.arange(-0.5, 0.6, 0.5)
    X1 = np.array(np.meshgrid(grid2, grid2)).T.reshape(len(grid2) ** 2, 2)

    # compute features for invasive measurements
    A_obs = []
    for x in X1:
        # shift X1 by x
        # for each element in X1 + x, compute the feature
        A_obs.append(np.array([phi(y, X0) for y in X1 + x]))

    X_obs = np.zeros((len(A_obs), d))
    A_obs = np.array(A_obs)

    A_int = []
    X_int = []
    # compute featurs for integrated measurements
    for x in X1:
        # shift X1 by x
        # for each element in X1 + x, compute the features for each point
        # then sum up the features over the points in X1 + x to "integrate" the reward
        features = np.sum(np.array([phi(y, X0) for y in X1 + x]), axis=0).reshape(1, -1)
        features /= len(X1)
        X_int.append(features.flatten())

        if indirect:
            A_int.append(np.zeros((len(X1), d)))
        else:
            # append zero observations to have same output dimension
            features = np.vstack([features, np.zeros((len(X1) - 1, d))])
            A_int.append(features)

    X_int = np.array(X_int)
    A_int = np.array(A_int)

    X = np.vstack([X_obs, X_int])
    A = np.vstack([A_obs, A_int])

    # set features s.t. the true function is exp(-x^2)
    theta = np.zeros(d)
    theta[18] = 1.

    # create game and instance
    _id = "laser"
    if indirect:
        _id += "-indirect"

    game = GenericPM(X, A, id=_id)
    instance = GameInstance(game, theta, noise_normal)

    return game, instance

# list of available games
GAMES = [simple_bandit, laser, counter_example]


def ucb(game_, **params):
    """
    strategy factory for UCB
    """
    lls = RegularizedLeastSquares(d=game_.get_d())
    estimator = RegretEstimator(game=game_, lls=lls, delta=0.05, truncate=False)
    strategy = UCB(game_, estimator=estimator)
    return strategy


def ids(game_, **params):
    """
    strategy factory for IDS
    params: --infogain {full,directed2,...} --dids
    """
    infogain_dict = dict([(f.__name__, f) for f in INFOGAIN])
    infogain = infogain_dict[params.get('infogain', 'full')]
    dids = params.get('dids')
    anytime = params.get('anytime')

    lls = RegularizedLeastSquares(d=game_.get_d())
    estimator = RegretEstimator(game=game_, lls=lls, delta=0.05, truncate=True)
    strategy = IDS(game_, infogain=infogain, estimator=estimator, deterministic=dids, anytime=anytime)
    return strategy


# list of available strategies
STRATEGIES = [ucb, ids]
# list of available info gains for IDS
INFOGAIN = [full, directed2, directed3, directeducb, info_game]


def run(game_factory, strategy_factory, **params):
    """
    run a game
    """
    # setup game and instance
    game, instance = game_factory(**params)

    # setup strategy
    strategy = strategy_factory(game, **params)

    # other parameters
    n = params.get('n')
    # lb_return = params.get('lb_return')

    outdir = params.get('outdir')
    if not os.path.exists(outdir):
        if query_yes_no(f"The target directory {outdir} does not exist. Do you want to create it?"):
            os.makedirs(outdir)
        else:
            print("Nothing written. Exiting.")
            exit()

    outdir = os.path.join(outdir, f"{game.id()}-{n}", instance.id(), strategy.id())

    if os.path.exists(os.path.join(outdir, 'params.json')) and params['overwrite']:
        shutil.rmtree(outdir)

    del params['overwrite']

    # setup output directory
    os.makedirs(outdir, exist_ok=True)
    if not os.path.exists(os.path.join(outdir, 'params.json')):
        # save params
        with open(os.path.join(outdir, 'params.json'), 'w') as outfile:
            json.dump(params, outfile)
        json.dumps(params)

    else:
        # check if directory has same parameters
        with open(os.path.join(outdir, 'params.json'), 'r') as file:
            prev_params = json.load(file)
            if not prev_params == params:
                if not query_yes_no(
                        f"WARNING: Input parameters changed. The previous parameters were:\n{prev_params}\n\n The current parameters are:\n{params}\n\nDo you want to continue?",
                        default="no"):
                    print("Nothing written. Exiting.")
                    exit()

    outfile = os.path.join(outdir, f"run-{timestamp()}-{uuid.uuid4().hex}.csv")


    data = np.zeros(shape=(n, 3))  # store data for the csv file
    cumulative_regret = 0

    # run game
    for t in range(n):
        # call strategy
        x = [strategy.get_next_action()]

        # compute reward and regret
        reward = instance.get_reward(x)
        regret = instance.get_max_reward() - reward
        cumulative_regret += regret

        # store data : regret, cumulative regret, index of arm pulled
        data[t] = regret, cumulative_regret, str(x)

        # get observation and update strategy
        observation = instance.get_noisy_observation(x)
        strategy.add_observations(x, observation)



    # outfile
    np.savetxt(outfile, data)

    return outdir


def main():
    # store available games and strategies as a dict
    games = dict([(f.__name__, f) for f in GAMES])
    strategies = dict([(f.__name__, f) for f in STRATEGIES])

    # setup argument parse
    parser = argparse.ArgumentParser(description='run a partial monitoring game.')
    parser.add_argument('game', choices=games.keys())
    parser.add_argument('strategy', choices=strategies.keys())
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--outdir', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--rep', type=int, default=1)
    parser.add_argument('--infogain', choices=[f.__name__ for f in INFOGAIN])
    parser.add_argument('--dids', action='store_true')
    parser.add_argument('--anytime', action='store_true', default=False)
    parser.add_argument('--laser-indirect', action='store_true', default=False)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--aggr', choices=[f.__name__ for f in aggregate.AGGREGATORS])
    parser.add_argument('--lb_return', type=bool, default=False)

    # parse arguments
    args = vars(parser.parse_args())

    # create game and strategy factories
    game_factory = games[args['game']]
    strategy_factory = strategies[args['strategy']]

    # repetition number
    rep = args['rep']
    del args['rep']

    # aggregation flag
    aggr = args['aggr']
    del args['aggr']


    lb_return = args['lb_return']


    # run game, possibly multiple times
    for i in range(rep):
        if rep > 1:
            print(f"Running iteration {i}.")
        path = run(game_factory, strategy_factory, **args)

    if lb_return:
        n = args['n']
        game, instance = game_factory(**args)
        outdir = args['outdir']
        outdir = os.path.join(outdir, f"{game.id()}-{n}", instance.id(), 'lb')
        # setup output directory
        os.makedirs(outdir, exist_ok=True)
        outfile = os.path.join(outdir, f"lb.csv")
        lb = lower_bound(game,instance)
        lbn = np.log(np.array(range(n))+1) * lb
        np.savetxt(outfile, lbn)

    # aggregate if requested pm2 laser ids --n=10000 --outdir=pm-runs/ --infogain=full
    if aggr:
        aggregator = aggregate.AGGREGATORS[[f.__name__ for f in aggregate.AGGREGATORS] == aggr]
        aggregate.aggregate(path, aggregator)


if __name__ == "__main__":
    main()
