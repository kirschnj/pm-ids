# Instructions

Install the package with `pip install -e .`

Example usage:

`pm2 simple_bandit --seed=3 ids --n=1000 --outdir=runs/ --ids_info=WorstCaseInfoGain --ids_gap=ValueGap`

`pm2 simple_bandit --seed=3 ucb --n=1000 --outdir=runs/`


In this case:
* `laser` is another environment. There is also `contextual_simple_bandit` and the end of optimism example `eoo`
* `ids` is the algorithm which can be configured using different arguments (`--ids_info`, `--ids_gap`). There is also `ucb`, `ts`
* `--n=1000` is the horizon

This puts performance data in `runs/` . The structure is `runs/env/algo-X-hash/run-hash.csv`. You can collect multiple runs for plotting later. For plotting, the runs need to be aggregated first. To do so, run:

`pm2-aggr runs/env-X/ regret`

This will compute aggregated statistics (mean, std-error) for the runs. The command goes recursively through directories, so you can also pass `pm2-aggr runs/ regret` to aggregate all environments. To do the actual plotting, make a copy of the notebook `notebooks/QuickPlotting.ipynb` and point the path to the environment you want to plot.

See `pm2 --help` and pm/main.py for more options and details


## Code
The main structure is:

* `game.py` defines the action & observation features. Examples in `pm/games/`
* `instance.py` defines the game parameter, so game+instance gives you one specific bandit model
* `strategy.py` defines the algorithm. Examples in `pm/strategies/`
* `estimator.py` is for parameter estimation, so least squares here, and an estimator to compute the gaps
