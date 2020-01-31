import argparse
import glob
import os
import numpy as np

def regret(data):
    regret = data[:,1,:]
    regret_avg = np.mean(regret, axis=1)
    regret_std = np.std(regret, axis=1)
    # regret_err = regret_std / np.sqrt(len(data))
    return np.vstack([regret_avg, regret_std]).T

AGGREGATORS = [regret]


def aggregate(path, aggregator):
    if not os.path.exists(os.path.join(path, 'params.json')):
        print(f"Directory {path} does not contain 'params.json'. Skipping.")
        return

        # list csv files
    csv_files = glob.glob(os.path.join(path, 'run-*.csv'))

    # no csv files, skip
    if len(csv_files) == 0:
        print(f"Directory {path} does not contain .csv files. Skipping.")
        return

    aggr_file = os.path.join(path, f'aggr-{aggregator.__name__}-{len(csv_files)}.csv')

    if os.path.exists(aggr_file):
        print(f"Aggregated file {aggr_file} exists. Skipping.")
        return

    # load first file to get length
    csv_data_0 = np.loadtxt(csv_files[0])
    data = np.empty(shape=(*csv_data_0.shape, len(csv_files)))

    # go through all csv files and store data
    for i, file in enumerate(csv_files):
        data[:, :, i] = np.loadtxt(file)

    aggr_data = aggregator(data)

    # save in csv file
    np.savetxt(aggr_file, aggr_data)
    print(f"Saved {aggr_file}")


def main():

    # store available games and strategies as a dict
    aggregators = dict([(f.__name__, f) for f in AGGREGATORS])

    # setup argument parse
    parser = argparse.ArgumentParser(description='run a partial monitoring game.')
    parser.add_argument('path')
    parser.add_argument('aggregator', choices=aggregators.keys())
    # parser.add_argument('--n', type=int, required=True)
    # parser.add_argument('--outdir', type=str)
    # parser.add_argument('--seed', type=int)
    # parser.add_argument('--infogain', choices=[f.__name__ for f in INFOGAIN])
    # parser.add_argument('--dids', action='store_true')

    # parse arguments
    args = vars(parser.parse_args())
    aggregator = aggregators[args['aggregator']]

    # run aggregation

    # list all directories that have a params.json
    for path in glob.iglob(os.path.join(args['path'], '**/'), recursive=True):
        aggregate(path, aggregator)


if __name__ == "__main__":
    main()
