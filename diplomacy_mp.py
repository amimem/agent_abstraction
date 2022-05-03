
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
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed


def get_game_tuple_presence(game_df, game_id):
    game_tuple_presence = {}
    # lens = []
    if game_df.unique_unit_id.nunique() == game_df.unique_unit_id.max():
        try:
            tuples = utils.get_n_tuples(game_df)
            random_tuples = utils.get_random_samples(tuples, 5)
        except Exception as msg:
            print(msg, flush=True)
        # tuple_lens = [len(tuples[n]) for n in tuples]
        # lens.append(tuple_lens)
        try:
            ret_dict = utils.get_tuples_presence(game_df, random_tuples)
            game_tuple_presence[game_id] = ret_dict
        except Exception as msg:
            print(msg, flush=True)
    return game_tuple_presence

def get_rows(game_tuple_presence):
    all_records = []
    for row in utils.gen_tuple_rows(game_tuple_presence):
        all_records.append(row.copy())
    return all_records
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', metavar='P', type=str, help='path')
    parser.add_argument('--dataset', metavar='D', type=str, help='path')
    parser.add_argument('--start', metavar='S', type=int, help='start')
    parser.add_argument('--end', metavar='E', type=int, help='end')
    parser.add_argument('--step', metavar='ST', type=int, help='step')
    args = parser.parse_args()

    # jobid = 0
    # os.getenv('SLURM_ARRAY_TASK_ID')
    # start_index = (int(jobid))*100
    
    step_size = args.step
    start_index = np.arange(args.start, args.end, step_size)
    path = os.path.join(args.path, args.dataset)
    num_games = len(start_index)
    pr_max_cpus = multiprocessing.cpu_count()
    th_max_cpus =  min(32, os.cpu_count() + 4)
    print("max_thread and max_process: ", pr_max_cpus, th_max_cpus, flush=True)
    # games_jsons = utils.load_jsonl(path, start_index=start_index, num_games=num_games, mmap=False, completed_only=True)

    games_jsons = []
    with ThreadPoolExecutor(max_workers=th_max_cpus) as executor:
        results = executor.map(utils.load_jsonl, [path]*num_games, [step_size]*num_games, [False]*num_games, start_index, [True]*num_games)

    for result in results:
        games_jsons.extend(result)

    # Convert to a pandas dataframe
    df = pd.DataFrame(games_jsons)

    games = df['phases']
    games.apply(lambda x: x[0])

    for game in games:
        for ix, iy in enumerate(game):
            game[ix]['phase_id'] = ix

    all_records = []
    for idx, game in enumerate(games):
        for idx, phase in enumerate(game):
            row_generator = utils.flatten_json(phase)
            assert row_generator is not None, row_generator
            for row in row_generator:
                all_records.append(row.copy())

    complete_df = pd.DataFrame.from_records(all_records)
    complete_df['unique_unit_id'] = -1

    unique_games = complete_df["game_id"].unique()

    game_phase_df_list = []
    for idx, game_id in enumerate(unique_games):
        phases_df_list = []
        print(idx, game_id, flush=True)
        s_dict = {}
        d_dict = {}
        _id = 1
        game_df = complete_df.loc[complete_df["game_id"].apply(lambda x: x == game_id)]
        unique_phases = game_df['phase_id'].unique()
        for phase in unique_phases:
            condition = game_df["phase_id"].apply(lambda x: x == phase)
            phase_df = game_df.loc[condition]
            s_dict, d_dict, _id = utils.assign_unit_id(phase_df, s_dict, d_dict, _id)
            phases_df_list.append(phase_df)
        phases_cdf = pd.concat(phases_df_list)
        dislodged_df = phases_cdf.loc[phases_cdf['action'] == -2].copy()
        utils.replace_dislodged_units(phases_cdf, dislodged_df)
        game_phase_df_list.append(phases_cdf)


    # cdf = pd.concat(game_phase_df_list, ignore_index=True)
    cdf = pd.concat(game_phase_df_list)
    assert cdf.loc[cdf['coordinator'] == 'RA'].empty

    spring_fall_phases=(cdf['phase_name'].apply(lambda x:x[0])!='W') & (cdf['phase_name'].apply(lambda x:x[-1])!='R') & (cdf['phase_name'].apply(lambda x:x[-1]) == 'M')
    cdf_sf = cdf.loc[spring_fall_phases].copy()
    cdf_sf['phase_num']=cdf_sf.phase_name.apply(lambda x: float(x[1:-1]+('.0' if x[0]=='S' else '.5')))

    game_df_array = []
    for idx, game_id in enumerate(unique_games):
        assert type(game_id) is str, (game_id, "is not a string")
        print(idx, game_id, flush=True)
        game_df = cdf_sf.loc[cdf_sf['game_id'] == game_id]
        game_df_array.append(game_df)

    with ProcessPoolExecutor(max_workers=pr_max_cpus) as executor:
        results = executor.map(get_game_tuple_presence, game_df_array, unique_games)
        game_tuple_presence_list = list(results)

    game_tuple_presence_dict = {}
    for game_tuple_presence in game_tuple_presence_list:
        # try:
        game_tuple_presence_dict.update(game_tuple_presence)
        # except Exception as msg:
        #     print(msg, "hello")

    
    all_records = []
    for row in utils.gen_tuple_rows(game_tuple_presence_dict):
        all_records.append(row.copy())
    
    df = pd.DataFrame.from_records(all_records)
    df.to_hdf(f'{args.path}/game_tuple_presence_{start_index[0]}_{start_index[-1]}_{num_games}.h5', key='df', mode='w')
    
    # with open('game_tiple_presence_dict.json', 'w') as f:
    #     json.dump(game_tiple_presence_dict, f)
    
    # with open(f'{args.path}/game_tiple_presence_{start_index}_{start_index+num_games}_{jobid}.json', 'w') as file:          
    #     json.dump(game_tiple_presence, file, indent=4)


if __name__ == "__main__":
    main()
