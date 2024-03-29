import argparse
import glob
import os
import numpy as np
import pandas as pd
import gc

def regret(data):
    """
    aggregator for cumulative regret (2nd column of each csv)
    """
    regret = data[:, 1, :]
    regret_avg = np.mean(regret, axis=1)
    regret_std = np.std(regret, axis=1)
    return np.vstack([regret_avg, regret_std]).T

def time(data):
    print(data)
    time_avg = np.mean(data, axis=-1)
    time_std = np.std(data, axis=-1)
    return np.vstack([time_avg, time_std]).T

def allocation(data):
    """
    aggregator for averaged allocations (third column of each csv)
    """
    allocations = data[:,2,:]
    # K = np.unique(allocations[0,:])).size
    allocations_avg = np.apply_along_axis(lambda x:np.histogram(x, bins=K, density=True)[0],0, allocations)

    return allocations_avg

AGGREGATORS = [regret, allocation, time]


def aggregate(path, aggregator, remove_old=False, file='run'):
    if not os.path.exists(os.path.join(path, 'params.json')):
        print(f"Directory {path} does not contain 'params.json'. Skipping.")
        return

        # list csv files
    csv_files = glob.glob(os.path.join(path, f'{file}-*.csv'))

    # no csv files, skip
    if len(csv_files) == 0:
        print(f"Directory {path} does not contain .csv files. Skipping.")
        return

    aggr_file = os.path.join(path, f'aggr-{aggregator.__name__}-{len(csv_files)}.csv')

    if remove_old:
        old_csv_files = glob.glob(os.path.join(path, f'aggr-{aggregator.__name__}-*.csv'))
        for old_file in old_csv_files:
            if old_file != aggr_file:
                os.remove(old_file)
                print(f"Removed {old_file}")

    if os.path.exists(aggr_file):
        print(f"Aggregated file {aggr_file} exists. Skipping.")
        return



    # load first file to get length
    csv_data_0 = pd.read_csv(csv_files[0], delimiter=" ", header=None)
    data = np.empty(shape=(*csv_data_0.shape, len(csv_files)))
    # go through all csv files and store data
    print(f"Reading {len(csv_files)} files ...")
    for i, file in enumerate(csv_files):
        # data[:, :, i] = np.loadtxt(file)
        # pandas is A LOT faster
        data[:, :, i] = pd.read_csv(file, delimiter=" ", header=None).values

    print("Files loaded. Aggregating now...")
    aggr_data = aggregator(data)

    # save in csv file
    np.savetxt(aggr_file, aggr_data)
    print(f"Saved {aggr_file}")
    gc.collect()


def main():
    # store available aggregator
    aggregators = dict([(f.__name__, f) for f in AGGREGATORS])

    # setup argument parser
    parser = argparse.ArgumentParser(description='run a partial monitoring game.')
    parser.add_argument('path')
    parser.add_argument('aggregator', choices=aggregators.keys())
    parser.add_argument('--remove_old', action='store_true')
    parser.add_argument('--file', type=str, default='run')

    # parse arguments
    args = vars(parser.parse_args())
    aggregator = aggregators[args['aggregator']]

    # run aggregation
    for path in glob.iglob(os.path.join(args['path'], '**/'), recursive=True):
        aggregate(path, aggregator, remove_old=args['remove_old'], file=args.get('file'))


if __name__ == "__main__":
    main()
