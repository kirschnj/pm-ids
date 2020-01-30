import argparse

import numpy as np
import os
import json
import uuid

from pm.estimator import RegularizedLeastSquares, RegretEstimator
from pm.game import GameInstance
from pm.games.bandit import Bandit
from pm.strategies.ucb import UCB
from pm.utils import query_yes_no, timestamp


def simple_bandit(**params):
    seed = params.get
    X = np.random.normal(size=12).reshape(6, 2)
    game = Bandit(X, id="simple_bandit")
    noise = lambda size : np.random.normal(0, 1, size=size)
    instance = GameInstance(game, theta=np.array([1., 0.]), noise=noise)

    return game, instance

GAMES = [simple_bandit]

def ucb(game_, **params):
    lls = RegularizedLeastSquares(d=game_.get_d())
    estimator = RegretEstimator(game=game_, lls=lls, delta=0.5, truncate=False)
    strategy = UCB(game_, estimator=estimator)
    return strategy

STRATEGIES = [ucb]


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

    outdir = params.get('outdir')
    if not os.path.exists(outdir):
        if query_yes_no(f"The target directory {outdir} does not exist. Do you want to create it?"):
            os.makedirs(outdir)
        else:
            print("Nothing written. Exiting.")
            exit()

    outdir = os.path.join(outdir, game.id(), instance.id(), strategy.id())


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
                if not query_yes_no(f"WARNING: Input parameters changed. The previous parameters were:\n{prev_params}\n\n The current parameters are:\n{params}\n\nDo you want to continue?", default="no"):
                    print("Nothing written. Exiting.")
                    exit()


    outfile = os.path.join(outdir, f"{timestamp()}-{uuid.uuid4().hex}.csv" )

    data = np.zeros(shape=(n, 2))  # store data for the csv file
    cumulative_regret = 0

    # run game
    for t in range(n):
        # call strategy
        x = strategy.get_next_action()

        # compute reward and regret
        reward = instance.get_reward(x)
        regret = instance.get_max_reward() - reward
        cumulative_regret += regret

        # store data
        data[t] = regret, cumulative_regret

        # get observation and update strategy
        observation = instance.get_noisy_observation(x)
        strategy.add_observations(x, observation)


    # outfile
    np.savetxt(outfile, data)


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

    # parse arguments
    args = vars(parser.parse_args())

    # create game and strategy factories
    game_factory = games[args['game']]
    strategy_factory = strategies[args['strategy']]

    # run game
    run(game_factory, strategy_factory, **args)

if __name__ == "__main__":
    main()
