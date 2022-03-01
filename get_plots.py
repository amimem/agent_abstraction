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



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', metavar='P', type=str, help='path')
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

    utils.plot(cdf, args.path)