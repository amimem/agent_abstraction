import json
import pandas as pd
import numpy as np
import mmap
import json
import itertools
import os
import json
import argparse
from lib.utils import utils

def dummy_func(x):
    counts_df = x.diff_sign.value_counts()
    return counts_df[1]/(counts_df[1] + counts_df[-1])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', metavar='P', type=str, help='path')
    parser.add_argument('--plot', metavar='PL', type=str, help='plot')
    args = parser.parse_args()

    start_index = 0
    end_index = 100

    h5_array = []

    for idx in range(start_index, end_index):
        path = f"{args.path}/hdf5/game_tiple_presence_{idx*100}_{(idx+1)*100}_{idx}.h5"

        try:
            # with open(f'{path}', 'r') as file:
            #     game_triple_presence = json.load(file)
            h5_array.append(pd.read_hdf(path, key='df'))
            print("appended", path)

        except Exception as e:
            print("path does not exist", path, e)

    cdf = pd.concat(h5_array)
    del h5_array



    if args.plot == "hist":
        utils.plot(cdf, args.path)
    elif args.plot == "accuracy":
        cdf['diff'] = cdf['factor_same'] - cdf['factor_diff']
        cdf['diff_sign'] = np.sign(cdf['diff'])
        gp_df = cdf.groupby(['min_phase_num', 'max_min_diff']).apply(dummy_func)
        utils.accuracy_plot(gp_df, args.path)
