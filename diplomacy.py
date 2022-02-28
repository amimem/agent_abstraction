
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
    parser.add_argument('--dataset', metavar='D', type=str, help='path')
    args = parser.parse_args()

    jobid = os.getenv('SLURM_ARRAY_TASK_ID')
    start_index = (int(jobid))*100
    num_games = 100
    path = os.path.join(args.path, args.dataset)
    games_jsons = utils.load_jsonl(path, start_index=start_index, num_games=num_games, mmap=False, completed_only=True)

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

    game_tiple_presence = {}
    for idx, game_id in enumerate(unique_games):
        assert type(game_id) is str, (game_id, "is not a string")
        print(idx, game_id, flush=True)
        game_df = cdf_sf.loc[cdf_sf['game_id'] == game_id]
        
        if game_df.unique_unit_id.nunique() == game_df.unique_unit_id.max():
            triples = utils.get_triples(game_df)
            try:
                emp = utils.get_triples_presence(game_df, triples)
                print(emp, len(triples), flush=True)
                game_tiple_presence[game_id] = triples
            except AssertionError as msg:
                print(msg, flush=True)
    
    with open(f'{args.path}/game_tiple_presence_{start_index}_{start_index+num_games}_{jobid}.json', 'w') as file:          
        json.dump(game_tiple_presence, file, indent=4)