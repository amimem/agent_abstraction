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

    game_df_array = []

    for idx in range(start_index, end_index):
        path = f"{args.path}/jsons/game_tiple_presence_{start_index*100}_{(start_index+1)*100}_{start_index}.json"
        with open(f'{path}', 'r') as file:
            game_triple_presence = json.load(file)

        all_records = []
        for row in utils.gen_triple_rows(game_triple_presence):
            all_records.append(row.copy())

        df = pd.DataFrame.from_records(all_records)
        df = df.dropna()

        game_df_array.append(df)

    cdf = pd.concat(game_df_array)
    del game_df_array

    utils.plot(cdf, args.path)