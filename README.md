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
