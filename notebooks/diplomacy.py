
import json
import pandas as pd
import numpy as np
import mmap
import json
import itertools
import os
import json
import argparse

# Load the json data
def load_jsonl(path: str, num_games: int = 30, mmap: bool = False, start_index = 0, completed_only: bool = False):
    '''
    Loads the jsonl data from the given path.
    If num_games is not -1, only the first num_games games are loaded.
    If mmap is True, the data is memory mapped.
    If completed_only is True, only completed games are loaded.
    '''
    games_jsons = []
    counter = 0
    with open(path, "r+b") as json_file:
        if mmap:
            with mmap.mmap(json_file.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_object:
                for i, line in enumerate(iter(mmap_object.readline, b"")):
                    tmp = json.loads(line.decode("utf-8"))
                    if tmp["map"] == "standard":
                        if completed_only:
                            for phase in tmp["phases"]:
                                if phase["name"] == "COMPLETED":
                                    counter += 1
                                    if counter > start_index:
                                        games_jsons.append(tmp)
                        else:
                            counter += 1
                            if counter > start_index:
                                games_jsons.append(tmp)

                    if num_games != -1 and len(games_jsons) == num_games:
                        print("last game line is", i, counter)
                        break
        else:
            for i, line in enumerate(json_file):
                tmp = json.loads(line.decode("utf-8"))
                if tmp["map"] == "standard":
                    if completed_only:
                        for phase in tmp["phases"]:
                            if phase["name"] == "COMPLETED":
                                counter += 1
                                if counter > start_index:
                                    games_jsons.append(tmp)
                    else:
                        counter += 1
                        if counter > start_index:
                                games_jsons.append(tmp)

                if num_games != -1 and len(games_jsons) == num_games:
                    print("last game line is", i, counter)
                    break

    return games_jsons


def flatten_json(input):
    out = {}
    out['game_id'] = input['state']['game_id']
    out['phase_id'] = input['phase_id']
    out['phase_name'] = input['name']
    results_units_keys = [x for x in input['results']]

    assert input['orders'].items() 
    for player, orders in input['orders'].items():
        if orders is not None:
            if orders:
                for order in orders:
                    out['coordinator'] = player
                    out['type'] = order.split()[0]
                    out['current_location'] = order.split()[1]
                    out['action'] = order.split()[2]
                    unit = order.split()[0]+ ' ' + order.split()[1]
                    if input['results']:
                        out['results'] = input['results'][unit]
                    else:
                        print("empty results")
                        print(order)
                    out['impact_location'] = []
                    if out['action'] == '-' or out['action'] == 'R':
                        assert len(order.split()) == 4 or len(order.split()) == 5, order
                        out['impact_location'] = order.split()[3]
                    if unit in results_units_keys:
                        results_units_keys.remove(unit)
                    yield(out)
                
            else:
                # uncmomment these if you need a row for empty orders (in group by you will get 1 instead of 0)
                # out['coordinator'] = player
                # out['type'] = -1
                # out['current_location'] = -1
                # out['action'] = -1
                # out['results'] = -1
                # out['impact_location'] = -1
                # if out['action'] == '-' or out['action'] == 'R':
                #     assert len(order.split()) == 4 or len(order.split()) == 5, order
                #     out['impact_location'] = order.split()[3]
                # yield(out)
                pass

    if len(results_units_keys)>0:
        results_units_values = [input['results'][x] for x in results_units_keys]
        total_len = 0
        is_movement = True if out['phase_name'][-1] == 'M' else False
        for l in results_units_values:
                total_len += len(l)
        # if we are in the last movement phase, the results are not meaningful
        for unit in results_units_keys:
            if unit == 'WAIVE': ## FIXME: double check this later
                continue
            assert len(unit.split()) == 2, unit
            if not (total_len == 0 and is_movement):
                out['coordinator'] = 'RA'
            else:
                for player in input["state"]["units"]:
                    if unit in input["state"]["units"][player]:
                        out['coordinator'] = player

            if len(unit.split()[0]) != 1: # if we have sth like HOL D: "void"
                assert input['results'][unit][0] == 'void', input['results'][unit]
                location = unit.split()[0]
                assert unit.split()[1] == 'D', unit.split()[1]
                # find the corresponding key in the results
                for key in input['results']:
                    if key.split()[1] == location:
                        assert input['results'][key][0] == "disband", input['results'][key]
                        input['results'][key].append(input['results'][unit][0])
                        break
            else:
                out['type'] = unit.split()[0]
                out['current_location'] = unit.split()[1]
                out['results'] = input['results'][unit]
                if not (total_len == 0 and is_movement):
                    out['action'] = -2 # not in orders and only in result // only happens for 
                else:
                    out['action'] = -3
                out['impact_location'] = -2
                yield(out)


def assign_unit_id(phase_df, source_unit_id_map, dest_unit_id_map, _id, discard_disband_creation=False):

    # fror each row in the phase df
    for idx, row in phase_df.iterrows():

        # if current location or type of army in invalid skip the row (we only deal with valid orders)
        if row['action'] == -1 or row['action'] == -2 or row['type'] == 'N':
            continue
        
        # get the current location of the unit
        source_unit = (row['type'] + ' ' + row['current_location'], row['coordinator'])

        # if the location is not in the map, add it to the map (in other phases the same unit can be used, hence checking the condition _ dictionaries are global, have data across phases)
        if source_unit not in source_unit_id_map:
            # if discard_disband_creation:
                # if row['action'] == 'D':
                #     print("disbanding a unit that does not exist", row)
                    # continue
            source_unit_id_map[source_unit] = _id
            _id += 1

        # destination dict is synced with source dict after the loop, so that we can use updated info at the beginning of each assignment
        phase_df.loc[idx,'unique_unit_id'] = source_unit_id_map[source_unit]

        if row['action'] == '-':
            result = row['results']
            if isinstance(result, list):
                if len(result) == 0:
                    try:
                        dest_unit = (row['type'] + ' ' + row['impact_location'], row['coordinator'])
                    except:
                        print("dest location error", row)
                        return
                    if dest_unit not in dest_unit_id_map:
                        dest_unit_id_map[dest_unit] = source_unit_id_map.pop(source_unit)
                elif 'disband' in result:
                    Exception("move with disband result")
                    # source_unit_id_map.pop(source_unit)
                    
        elif row['action'] == 'R':
            result = row['results']
            if isinstance(result, list):
                if len(result) == 0:
                    dest_unit = (row['type'] + ' ' + row['impact_location'], row['coordinator'])
                    if dest_unit not in dest_unit_id_map:
                        dest_unit_id_map[dest_unit] = source_unit_id_map.pop(source_unit)
                elif 'disband' in result:
                    if len(result) > 1:
                        if 'void' in result:
                                print(result)
                    source_unit_id_map.pop(source_unit)

        elif row['action'] == 'D':
            result = row['results']
            # if row.phase_id == 23:
            # print(result, "here")
            # print(result, type(result),isinstance(result, list), "no")
            assert source_unit in source_unit_id_map
            if isinstance(result, list):
                if len(result) == 0:
                    source_unit_id_map.pop(source_unit)
                    # del source_unit_id_map[source_unit]
                    # if row.phase_id == 23:
                    # print("popped it out")
                elif 'disband' in result:
                    source_unit_id_map.pop(source_unit)
                elif 'void' in result:
                    print("void disband", row)
            else:
                 print(result, type(result),isinstance(result, list), "yes")
        
        elif row['action'] == 'B':
            assert source_unit in source_unit_id_map

        elif row['action'] == 'H':
            assert source_unit in source_unit_id_map

        elif row['action'] == 'S':
            assert source_unit in source_unit_id_map
        
        elif row['action'] == 'C':
            assert source_unit in source_unit_id_map

        # for added result rows
        elif row['action'] == -2:
            pass
            # result = row['results']
            # assert source_unit in source_unit_id_map
            # if isinstance(result, list):
            #     if len(result) == 0:
            #         source_unit_id_map.pop(source_unit)
            #     # elif 'disband' in result and 'void' not in result:
            #     #     source_unit_id_map.pop(source_unit)
            #     # elif 'disband' in result and 'void' in result:
            #     #     source_unit_id_map.pop(source_unit)
            #     elif 'disband' in result:
            #         source_unit_id_map.pop(source_unit)
            #     else:
            #         print("unknown result", result)

        elif row['action'] == -3:
            assert source_unit in source_unit_id_map

        else:
            print("invalid action", row)

    # merge the source and destination dictionaries into one
    source_unit_id_map.update(dest_unit_id_map)
    # remove the destination dict (values get updated based on old data if we don't do this)
    dest_unit_id_map = {}

    return source_unit_id_map, dest_unit_id_map, _id


def replace_dislodged_units(phases_cdf, dislodged_df):
    for idx, row in dislodged_df.iterrows():
        cond = phases_cdf["game_id"].apply(lambda x: x == row.game_id) & phases_cdf["type"].apply(lambda x: x == row.type) & phases_cdf["current_location"].apply(lambda x: x == row.current_location) & phases_cdf["phase_id"].apply(lambda x: x < row.phase_id) & phases_cdf["results"].apply(lambda x: 'dislodged' in x)
        assert phases_cdf.loc[cond].empty == False, (game_id, row)
        c = phases_cdf.loc[cond].iloc[-1]['coordinator']
        phases_cdf.loc[idx,'coordinator'] = c


#https://stackoverflow.com/questions/45655936/how-to-test-all-items-of-a-list-are-disjoint
# make sure some units are handed over to other players
def all_disjoint(iterables):
    merged = itertools.chain(*iterables)
    total = list(merged)
    total.sort()
    print(total)
    print(set(total))
    return len(total) == len(set(total))



def get_triples(game_df):
    players = game_df["coordinator"].unique()
    assert len(players) == 7, players
    player_units = {}
    for player in players:
        player_units[player] = game_df.loc[game_df["coordinator"] == player]["unique_unit_id"].unique()
    
    triples = []
    unique_two_players = set(itertools.permutations(player_units.keys(), 2))
    for player_i, player_j in unique_two_players:
        p_units = list(itertools.combinations(player_units[player_i], 2))
        o_unit = list(itertools.combinations(player_units[player_j], 1))
        # get all 3-tuples unique combinations
        p_o_product = list(itertools.product(p_units, o_unit))
        p_o_product = [(int(a), int(b), int(c)) for (a,b), (c,) in p_o_product]
        triple = [dict(triple=t, player_i=player_i, player_j=player_j) for t in p_o_product]
        triples.extend(triple)
    return triples


def get_triples_presence(game_df, triples):
    empty_eligible_phaes = 0
    for triple in triples:
        condition_i = (game_df["unique_unit_id"].apply(lambda x: x in triple['triple'][:2])) & (game_df["coordinator"].apply(lambda x: x == triple['player_i']))
        condition_j = (game_df["unique_unit_id"].apply(lambda x: x in triple['triple'][2:])) & (game_df["coordinator"].apply(lambda x: x == triple['player_j']))
        presence = game_df.loc[condition_i | condition_j]
        phases = presence["phase_name"].value_counts()
        mask = (phases == 3).to_dict()
        eligible_phases = presence.loc[presence["phase_name"].apply(lambda x: mask[x])]
        unqiue_eligible_phases = eligible_phases.phase_num.unique()
        max_phase_num = eligible_phases.phase_num.max()
        min_phase_num = eligible_phases.phase_num.min()
        max_min_diff = max_phase_num - min_phase_num
        if len(unqiue_eligible_phases):
            assert (max_min_diff/0.5 + 1) == len(unqiue_eligible_phases) , ("values are not contiguous", len(unqiue_eligible_phases), unqiue_eligible_phases, max_min_diff, game_df["game_id"].unique(), triple)
        else:
            empty_eligible_phaes += 1
        triple['max_phase_num'] = float(max_phase_num)
        triple['min_phase_num'] = float(min_phase_num)
        triple['max_min_diff'] = float(max_min_diff)
    return empty_eligible_phaes


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', metavar='P', type=str, help='path')
    parser.add_argument('--dataset', metavar='D', type=str, help='path')
    args = parser.parse_args()

    jobid = os.getenv('SLURM_ARRAY_TASK_ID')
    start_index = (int(jobid))*100
    num_games = 100
    path = os.path.join(args.path, args.dataset)
    games_jsons = load_jsonl(path, num_games=num_games, mmap=False, completed_only=True)

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
            row_generator = flatten_json(phase)
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
            s_dict, d_dict, _id = assign_unit_id(phase_df, s_dict, d_dict, _id, discard_disband_creation=False)
            phases_df_list.append(phase_df)
        phases_cdf = pd.concat(phases_df_list)
        dislodged_df = phases_cdf.loc[phases_cdf['action'] == -2].copy()
        replace_dislodged_units(phases_cdf, dislodged_df)
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
            triples = get_triples(game_df)
            try:
                emp = get_triples_presence(game_df, triples)
                print(emp, len(triples), flush=True)
                game_tiple_presence[game_id] = triples
            except AssertionError as msg:
                print(msg, flush=True)
    
    with open(f'{args.path}/game_tiple_presence_{start_index}_{start_index+num_games}_{jobid}.json', 'w') as file:          
        json.dump(game_tiple_presence, file, indent=4, sort_keys=True,)