
from cmath import nan
import json
import pandas as pd
import numpy as np
import mmap
import json
import itertools
import os
import json
import matplotlib.pyplot as plt
import time
import seaborn as sns

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
                        out['results'] = list(filter(None, out['results']))
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
                out['results'] = list(filter(None, out['results']))
                if not (total_len == 0 and is_movement):
                    out['action'] = -2 # not in orders and only in result // only happens for 
                else:
                    out['action'] = -3
                out['impact_location'] = -2
                yield(out)


def assign_unit_id(phase_df, source_unit_id_map, dest_unit_id_map, _id):

    # fror each row in the phase df
    for idx, row in phase_df.iterrows():

        # if current location or type of army in invalid skip the row (we only deal with valid orders)
        if row['action'] == -1 or row['action'] == -2 or row['type'] == 'N':
            continue
        
        # get the current location of the unit
        source_unit = (row['type'] + ' ' + row['current_location'], row['coordinator'])

        # if the location is not in the map, add it to the map (in other phases the same unit can be used, hence checking the condition _ dictionaries are global, have data across phases)
        if row['action'] == 'B':
            if source_unit in source_unit_id_map:
                print("build before disband", source_unit, source_unit_id_map[source_unit], _id)
                source_unit_id_map[source_unit] = _id 
                _id += 1

        if source_unit not in source_unit_id_map:
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
            assert source_unit in source_unit_id_map
            if isinstance(result, list):
                if len(result) == 0:
                    source_unit_id_map.pop(source_unit)
                    # del source_unit_id_map[source_unit]
                elif 'disband' in result:
                    source_unit_id_map.pop(source_unit)
                elif 'void' in result: # void means it is not the unit of the curent phase
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
        assert phases_cdf.loc[cond].empty == False, (row.game_id, row)
        c = phases_cdf.loc[cond].iloc[-1]
        phases_cdf.loc[idx,'coordinator'] = c.coordinator
        phases_cdf.loc[idx,'unique_unit_id'] = c.unique_unit_id


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

def get_n_tuples(game_df, n=5):
    players = game_df["coordinator"].unique()
    assert len(players) == 7, players
    player_units = {}
    for player in players:
        player_units[player] = game_df.loc[game_df["coordinator"] == player]["unique_unit_id"].unique()
    
    n_tuples = {}
    for player_i in players:
        n_tuples[player_i] = {}
        for k in range(n):
            n_tuples[player_i][k] = []
            values_i = player_units[player_i]
            if len(values_i) >= k+1:
                values_j = np.concatenate([value for key, value in player_units.items() if key != player_i])
                if len(values_j) >= n-k-1:
                    p_units = list(itertools.combinations(values_i, k+1))
                    o_unit = list(itertools.combinations(values_j, n-k-1))
                    # get all 3-tuples unique combinations
                    p_o_product = list(itertools.product(p_units, o_unit))
                    # p_o_product = [element for tupl in p_o_product for element in tupl]
                    # p_o_product = list(sum(p_o_product, ()))
                    n_tuples[player_i][k].extend(p_o_product)
    return n_tuples

def get_random_samples(n_tuples, n=5, num_samples_per_k = 1000):
    samples = []
    for player, value_dict in n_tuples.items():
        for k in value_dict:
            k_list = value_dict[k]
            if len(k_list) > 0:
                random_list_index = np.random.choice(len(k_list), num_samples_per_k)
                samples.extend([k_list[i] for i in random_list_index])
            else:
                print("no samples for", player, k)
    if len(samples) < n*num_samples_per_k*7:
        print("not enough samples", len(samples))
    return samples

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

            if max_min_diff:
                pivot_df = eligible_phases.groupby(['unique_unit_id','phase_num'])['action'].aggregate('first').unstack()
                tiple_array = np.array(triple['triple'])
                same_indice = tiple_array[[0,1]]
                diff_indice = tiple_array[[0,2]]

                joint_i = len(pivot_df.loc[same_indice].T.drop_duplicates())
                joint_j = len(pivot_df.loc[diff_indice].T.drop_duplicates())

                pivot_df['unit_counts'] = pivot_df.nunique(axis=1)

                prod_i = pivot_df.loc[same_indice, 'unit_counts'].values.prod()
                prod_j = pivot_df.loc[diff_indice, 'unit_counts'].values.prod()

                triple['factor_same'] = prod_i/ float(joint_i)
                triple['factor_diff'] = prod_j/ float(joint_j)
                triple['table'] = pivot_df.to_dict(orient='index')

        else:
            empty_eligible_phaes += 1
            
        triple['max_phase_num'] = float(max_phase_num)
        triple['min_phase_num'] = float(min_phase_num)
        triple['max_min_diff'] = float(max_min_diff)
    return empty_eligible_phaes

def get_tuples_presence(game_df, n_tuples, n=5):
    empty_eligible_phaes = 0
    counter = 0
    eligible_counter = 0
    return_list = []
    for tupl in n_tuples:
        condition_i = (game_df["unique_unit_id"].apply(lambda x: x in tupl[0]))
        condition_j = (game_df["unique_unit_id"].apply(lambda x: x in tupl[1]))
        presence = game_df.loc[condition_i | condition_j]
        phases = presence["phase_name"].value_counts()
        mask = (phases == n).to_dict()
        eligible_phases = presence.loc[presence["phase_name"].apply(lambda x: mask[x])]
        unqiue_eligible_phases = eligible_phases.phase_num.unique()
        counter += 1
        if len(unqiue_eligible_phases):
            max_phase_num = eligible_phases.phase_num.max()
            min_phase_num = eligible_phases.phase_num.min()
            max_min_diff = max_phase_num - min_phase_num
            
            assert (max_min_diff/0.5 + 1) == len(unqiue_eligible_phases) , ("values are not contiguous", len(unqiue_eligible_phases), unqiue_eligible_phases, max_min_diff, game_df["game_id"].unique(), tupl)

            if max_min_diff:
                pivot_df = eligible_phases.groupby(['unique_unit_id','phase_num'])['action'].aggregate('first').unstack()

                joint = len(pivot_df.T.drop_duplicates())
                unit_count = pivot_df.nunique(axis=1)

                d = {}
                d[0] = tupl
                d[1] = min_phase_num
                d[2] = max_min_diff
                d[3] = unit_count.to_list()
                d[4] = joint
                # d[5] = unit_count.values.prod()/ float(joint)
                # d[2] = pivot_df.to_dict(orient='index')
                return_list.append(d)

                eligible_counter += 1

        else:
            empty_eligible_phaes += 1

        if counter % 1000 == 0:
            print(counter, eligible_counter, empty_eligible_phaes)

    return return_list



def get_consecutive_elements(arr, k):
    return list(map(list, zip(*(arr[i:] for i in range(k)))))

def gen_triple_rows(game_tiple_presence):
    row = {}
    for game in game_tiple_presence:
        row['game_id'] = game
        for triple in game_tiple_presence[game]:
            row['player_i'] = triple['player_i']
            row['player_j'] = triple['player_j']
            row['min_phase_num'] = triple['min_phase_num']
            row['max_phase_num'] = triple['max_phase_num']
            row['max_min_diff'] = triple['max_min_diff']
            row['triple_0'] = triple['triple'][0]
            row['triple_1'] = triple['triple'][1]
            row['triple_2'] = triple['triple'][2]
            row['factor_same'] = triple['factor_same'] if 'factor_same' in triple else nan
            row['factor_diff'] = triple['factor_diff'] if 'factor_diff' in triple else nan
            yield row

def gen_tuple_rows(game_tiple_presence):
    row = {}
    for game in game_tiple_presence:
        row['game_id'] = game
        for tuple in game_tiple_presence[game]:
            row['tuple'] = tuple[0]
            row['min_phase_num'] = tuple[1]
            row['max_min_diff'] = tuple[2]
            row['unit_counts'] = tuple[3]   
            row['joint'] = tuple[4]
            row['k'] = len(tuple[0][0])
            yield row


def plot(df, path):

    window_vec=np.arange(0.5,3.5,0.5)
    fig,ax=plt.subplots(len(window_vec),3,figsize=(12,3*len(window_vec)))
    binvec=np.linspace(1,3,60)

    for wit,window_len in enumerate(window_vec):
        tmp_df = df.loc[df.max_min_diff == window_len, ['factor_same', 'factor_diff']]
        total_value_counts = tmp_df.value_counts()
        window_pair_index_values = total_value_counts.index.values
        window_pair_values = total_value_counts.values
        l1,l2=zip(*list(window_pair_index_values))

        ax[wit,0].scatter(l1,l2,s=list(window_pair_values*1000/window_pair_values.sum()))
        ax[wit,0].plot([binvec[0], binvec[-1]],[binvec[0],binvec[-1]],'k--')
        ax[wit,0].set_ylabel('different')
        ax[wit,0].set_xlabel('same')
        dstrvec=['same','different']
        counts,bins=np.histogram(tmp_df.factor_same,bins=binvec)
        ax[wit,1].plot(bins[:-1],counts,label=dstrvec[0])
        counts,bins=np.histogram(tmp_df.factor_diff,bins=binvec)
        ax[wit,1].plot(bins[:-1],counts,label=dstrvec[1])
        ax[wit,1].legend(frameon=False)
        ax[wit,1].set_title('window size '+str(int(window_len*2))+' steps')

        data=tmp_df.factor_same - tmp_df.factor_diff
        counts,bins=np.histogram(data,bins=np.linspace(-binvec[-1],binvec[-2],len(binvec)*2))
        ax[wit,2].plot(bins[:-1],counts)
        ax[wit,2].set_xlabel('same-different')
        ax[wit,2].plot([np.mean(data)]*2,ax[wit,2].get_ylim(),'k--')

    fig.tight_layout()
    save_path = f"{path}/same_diff_ver_windowsize_{time.time()}.pdf"
    fig.savefig(save_path,format='pdf',dpi=300,bbox_inches='tight')

def gen_triple_window_rows(game_tiple_presence):
    row = {}
    for game in game_tiple_presence:
        total_triples = len(game_tiple_presence[game])
        print(game, total_triples)
        for idx, triple in enumerate(game_tiple_presence[game]):
            if idx % 500 == 0:
                print(idx, total_triples, idx/total_triples)
            if triple['max_min_diff'] > 0:
                tiple_array = np.array(triple['triple'])
                table = triple['table']
                df = pd.DataFrame.from_dict(table, orient='index')
                df = df.drop('unit_counts', axis=1)
                df.index = df.index.astype(int)
                df.columns = df.columns.astype(float)
                column_list = df.columns.to_list()
                same_indice = tiple_array[[0,1]]
                diff_indice = tiple_array[[0,2]]
                
                consec_array = []
                for i in range(1, len(column_list)):
                    l = get_consecutive_elements(column_list, i+1)
                    consec_array.extend(l)

                for e in consec_array:
                    tmp_df = df.loc[:, e]
                    joint_i = len(tmp_df.loc[same_indice].T.drop_duplicates())
                    joint_j = len(tmp_df.loc[diff_indice].T.drop_duplicates())
                    counts_df = tmp_df.nunique(axis=1)
                    prod_i = counts_df[same_indice].values.prod()
                    prod_j = counts_df[diff_indice].values.prod()
                    factor_same = prod_i/ float(joint_i)
                    factor_diff = prod_j/ float(joint_j)

                    row['min_phase_num'] = min(e)
                    row['max_phase_num'] = max(e)
                    row['max_min_diff'] = max(e) - min(e)

                    row['factor_same'] = factor_same
                    row['factor_diff'] = factor_diff
                    yield row

def accuracy_plot(df, path):
    # make heatmap of the dataframe
    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(111)
    sns.heatmap(df.unstack().T, xticklabels= 6, yticklabels= 6, ax=ax)
    ax.set_xlabel("Year")
    ax.set_ylabel("Window Size, w")
    ax.set_title("Accuracy")
    x_labels = [str(x.get_text()[:-2]) for x in ax.get_xticklabels()]
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(f'{path}/heatmap.pdf_{time.time()}', dpi=300, bbox_inches='tight')