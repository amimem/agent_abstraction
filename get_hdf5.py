
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

    jobid = int(os.getenv('SLURM_ARRAY_TASK_ID'))

    j_path = f"{args.path}/jsons/game_tiple_presence_{jobid*100}_{(jobid+1)*100}_{jobid}.json"
    try:
        with open(f'{j_path}', 'r') as file:
            game_triple_presence = json.load(file)

        all_records = []
        for row in utils.gen_triple_window_rows(game_triple_presence):
            all_records.append(row.copy())

        df = pd.DataFrame.from_records(all_records)
        # df = df.dropna()
        # df = df.reset_index(drop=True)
        # df = df[["game_id", "max_min_diff", "factor_same", "factor_diff"]]
        print("appended", j_path)

        try:
            h_path = f"{args.path}/hdf5_3/game_tiple_presence_{jobid*100}_{(jobid+1)*100}_{jobid}.h5"
            df.to_hdf(h_path, key='df', mode='w')
            print("saved", h_path)
        except Exception as e:
            print("cannot create file", h_path, e)

    except:
        print("path does not exist", j_path)