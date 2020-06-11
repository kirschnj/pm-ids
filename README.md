# Instructions

Install the package with `pip install -e .`

Example usage:

`pm2 laser ids --n=10000 --outdir=pm-runs/ --infogain=full`

In this case:
* `laser` is the environment. There is also `simple_bandit`
* `ids` is the algorithm. There is also ucb
* `infogain` chooses the information gain for ids
* `--n=10000` is the horizon

See `pm2 --help` and pm/main.py for more options and details


## Code
The main structure is:

* `game.py` defines the action & observation features. Examples in `pm/games/`
* `instance.py` defines the game parameter, so game+instance gives you one specific bandit model
* `strategy.py` defines the algorithm. Examples in `pm/strategies/`
* `estimator.py` is for parameter estimation, so least squares here, and an estimator to compute the gaps
