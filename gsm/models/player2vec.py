import sys
import os
import pandas as pd
import numpy as np
import math
import random
import json
import copy
from typing import Tuple
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KDTree
from scipy.spatial import distance
sys.path.append(os.path.abspath(os.path.join('..', 'gsm')))
import gsm.conf.conf_data_prep as cfg_dp
import gsm.conf.conf_player2vec as cfg


# Model training

def load_list_and_map(infile: str, col: str) -> Tuple[list, dict]:
    """
    Load list and dict of elements in column col, for future reference (retrieving names etc.)
    
    Args:
        infile: input file
        col: dataframe column containing keys/names to read
        
    Returns:
        List, dict of elements (list for mapping by index, dict for mapping by name)
    """
    print(f'Running load_list_and_map on file {infile}...')
    elts_list = []
    elts_dict = {}
    df = pd.read_csv(infile)
    elts_list = df[col].to_list()
    for i in range(len(elts_list)):
        elts_dict[elts_list[i]] = i
    print(f'...found {len(elts_list)} elements')
    return elts_list, elts_dict


def infer_full_context(s_round: str, base_tourney_name: str, contexts_dict: dict) -> int:
    """
    Building full context name from round and tournament name, and get matching index
        
    Args:
        s_round: round in raw string form
        base_tourney_name: tournament name
        contexts_dict
        
    Returns:
        Index of input context in contexts list
    """
    i_round = -1
    if s_round in ['Q1R','Q2R','QQ','final round','qualifying round']:
        i_round = 0
    elif s_round in ['1R','round 1','2R','round 2','3R','round 3','3rd round','4R','round 4','4th round']:
        i_round = 1
    elif s_round in ['QF','quarterfinal','quarter finals','SF','semifinal','semi finals','F','final','final[1]']:
        i_round = 2
    if i_round>=0:
        s_context = f'{base_tourney_name}_{i_round}'
        i_context = contexts_dict[s_context]
        return i_context
    else:
        print(f'warning: could not infer stage of round {s_round}')
        return -1


def infer_winner_from_scoreline(scoreline: str) -> int:
    """
    Find winner by looking at the (last set) results
        
    Args:
        scoreline: final match scoreline, separated by set
        
    Returns:
        1 (2) if player one (two) was the winner, respectively
    """
    players_scores = scoreline[2:-2].split('], [')
    try:
        p_one_score_last_round = int(players_scores[0].split(', ')[-1])
    except ValueError:
        p_one_score_last_round = 0
    try:
        p_two_score_last_round = int(players_scores[1].split(', ')[-1])
    except ValueError:
        p_two_score_last_round = 0
    #print(f'{scoreline} => {p_one_score_last_round} | {p_two_score_last_round}')
    if p_one_score_last_round>p_two_score_last_round:
        return 1
    elif p_two_score_last_round>p_one_score_last_round:
        return 2
    else:
        #print(f'warning: found equal scoreline in {scoreline}')
        return 0


def update_edges(edges_dict: dict, ind_player: int, ind_context: int, wins: bool):
    """
    Helper function to update the edges dict depending on the result
    """
    p_dict = {}
    if ind_player in edges_dict:
        p_dict = edges_dict[ind_player]
    c_dict = {'wins': 0, 'losses': 0}
    if ind_context in p_dict:
        c_dict = p_dict[ind_context]
    if wins:
        c_dict['wins'] += 1
    else:
        c_dict['losses'] += 1
    p_dict[ind_context] = c_dict
    edges_dict[ind_player] = p_dict


def load_edges_and_players_from_results(players_feats_file: str, results_folder: str,
                                        contexts_dict: dict) -> Tuple[list, list, dict]:
    """
    Parse results from folder as weighted edges of the joint graph representation
    
    edge = [player,context,wins,losses]
    Also read players file, loading structures similar to 'load_list_and_map' only for players with valid results
        
    Args:
        players_feats_file: players info file
        results_folder: match results root folder
        contexts_dict: mapping str->int of contexts
        
    Returns:
        List of edges, list, dict of players
    """
    print('Running load_edges_and_players_from_results...')
    df_p = pd.read_csv(players_feats_file)
    p_names = df_p['name'].to_list()
    raw_players_dict = {n: 0 for n in p_names}
    edges_dict = {}
    dirs = [d for d in os.listdir(results_folder) if os.path.isdir(os.path.join(results_folder, d))]
    for d in dirs:
        subfolder = os.path.join(results_folder, d)
        files = [d for d in os.listdir(subfolder) if os.path.isfile(os.path.join(subfolder, d))]
        for f in files:
            f_path = os.path.join(subfolder, f)
            #print(f'...reading {f_path}')
            f = open(f_path, 'r')
            lines = f.readlines()
            header = lines[0]
            f.close()
            if len(lines)>3:
                header_dict = json.loads(header)
                base_tourney_name = header_dict['name']
                df = pd.read_csv(f_path, skiprows=1)
                for _, row in df.iterrows():
                    if row['player_one'] in raw_players_dict and row['player_two'] in raw_players_dict:
                        player_one = row['player_one']
                        player_two = row['player_two']
                        ind_context = infer_full_context(row['round'], base_tourney_name, contexts_dict)
                        i_winner = infer_winner_from_scoreline(row['score'])
                        # ignore matches where one player forfeited (don't consider this as a win)
                        if i_winner==1:
                            update_edges(edges_dict, player_one, ind_context, wins=True)
                            update_edges(edges_dict, player_two, ind_context, wins=False)
                            raw_players_dict[player_one] = 1
                            raw_players_dict[player_two] = 1
                        elif i_winner==2:
                            update_edges(edges_dict, player_one, ind_context, wins=False)
                            update_edges(edges_dict, player_two, ind_context, wins=True)
                            raw_players_dict[player_one] = 1
                            raw_players_dict[player_two] = 1
    # filter players dict
    players_dict = {}
    players_list = []
    i = 0
    for key in raw_players_dict:
        if raw_players_dict[key]>0:
            players_dict[key] = i
            players_list.append(key)
            i += 1
    weighted_edges_list = []
    for player in edges_dict:
        player_ind = players_dict[player]
        p_dict = edges_dict[player]
        for context in p_dict:
            wins = p_dict[context]['wins']
            losses = p_dict[context]['losses']
            weighted_edge = [player_ind,context,wins,losses]
            weighted_edges_list.append(weighted_edge)
    print(f'...found {len(weighted_edges_list)} edges')
    print(f'...found {len(players_dict)} | {len(players_list)} players with results')
    return weighted_edges_list, players_list, players_dict


def load_results(results_folder: str, players_dict: dict, contexts_dict: dict) -> list:
    """
    Lightweight version of 'load_edges_and_players_from_results', only loading results as edges
    
    edge = [player,context,wins,losses]
    Used to read test data for model evaluation

    Args:
        results_folder: match results root folder
        players_dict: mapping str->int of players
        contexts_dict: mapping str->int of contexts
        
    Returns:
        List of edges
    """
    match_results = []
    dirs = [d for d in os.listdir(results_folder) if os.path.isdir(os.path.join(results_folder, d))]
    for d in dirs:
        subfolder = os.path.join(results_folder, d)
        files = [d for d in os.listdir(subfolder) if os.path.isfile(os.path.join(subfolder, d))]
        for f in files:
            f_path = os.path.join(subfolder, f)
            f = open(f_path, 'r')
            lines = f.readlines()
            header = lines[0]
            f.close()
            if len(lines)>3:
                header_dict = json.loads(header)
                base_tourney_name = header_dict['name']
                df = pd.read_csv(f_path, skiprows=1)
                for _, row in df.iterrows():
                    if row['player_one'] in players_dict and row['player_two'] in players_dict:
                        ind_player_one = players_dict[row['player_one']]
                        ind_player_two = players_dict[row['player_two']]
                        ind_context = infer_full_context(row['round'], base_tourney_name, contexts_dict)
                        i_winner = infer_winner_from_scoreline(row['score'])
                        # ignore matches where one player forfeited (don't consider this as a win)
                        if i_winner>0:
                            res = [ind_player_one,ind_player_two,ind_context,i_winner]
                            match_results.append(res)
    return match_results


def sigmoid(x):
    sig = 1. / (1. + math.exp(-x))
    return sig


def get_random_embedding(rng: np.random, d: int, lower: float, upper: float) -> np.array:
    """
    Generates random embedding with values bounded

    Args:
        rng: numpy random generator
        d: number of dimensions for the embedding
        lower: lower bound of the values to generate
        upper: upper bound
        
    Returns:
        Random embedding
    """
    embd = [rng.uniform(lower,upper) for _ in range(d)]
    return np.array(embd)

    
def init_embeddings(n_elts: int, d: int, seed=16) -> list:
    """
    Generate n_elts random embeddings
    
    edge = [player,context,wins,losses]

    Args:
        n_elts: number of embeddings to generate
        d: number of dimensions
        seed: seed of the rng
        
    Returns:
        List of random embeddings
    """
    rng = np.random.default_rng(seed)
    lower_bound = -1./(2.*float(d))
    upper_bound = 1./(2.*float(d))
    x_p = []
    for _ in range(n_elts):
        emb = get_random_embedding(rng, d, lower_bound, upper_bound)
        x_p.append(emb)
    return x_p

def compute_embeddings_pairwise_relevance(embd_one: np.array, embd_two: np.array, method='norm') -> float:
    """
    Compute pairwise relevance (similarity) between two embeddings
    
    Use 'norm' as method

    Args:
        embd_one: embedding
        embd_two: embedding
        method = 'norm' or 'euclidean'
        
    Returns:
        Pairwise relevance
    """
    if method=='norm':
        rel = math.exp(-np.linalg.norm(embd_one-embd_two))
    elif method=='euclidean':
        rel = distance.euclidean(embd_one, embd_two)
    else:
        print(f'warning: invalid method {method} for compute_embeddings_pairwise_relevance; defaulting to norm')
        rel = math.exp(-np.linalg.norm(embd_one-embd_two))
    return rel

def meets_conditions(prev_rel: float, embd_one: np.array, embd_two: np.array, embd_three: np.array) -> bool:
    """
    Check that the three embeddings fulfill the condition for anchor embeddings generation

    Args:
        prev_rel: relevance between embd_one and embd_two
        embd_one: embedding
        embd_two: embedding
        embd_three: embedding
        
    Returns:
        List of edges
    """
    rel_one_three = compute_embeddings_pairwise_relevance(embd_one, embd_three)
    rel_two_three = compute_embeddings_pairwise_relevance(embd_two, embd_three)
    #print(f'meets_conditions : {prev_rel} | {rel_one_three} | {rel_two_three}')
    first_condition = (prev_rel > rel_one_three)
    second_condition = (prev_rel > rel_two_three)
    return (first_condition and second_condition)

def adjust_embd_manually(rng: np.random, embd_one: np.array, embd_two: np.array,
                         lower: float, upper: float, factor=0.1) -> np.array:
    """
    Adjust embedding values by hand to force it to meets the condition
    
    Does not really work; run more random attempts instead for now...

    Args:
        rng: random generator
        embd_one: embedding
        embd_two: embedding
        lower: lower bound
        upper: upper bound
        
    Returns:
        New embedding
    """
    embd = []
    for i in range(len(embd_one)):
        diff = abs(embd_one[i] - embd_two[i])
        step = diff*(1.+factor)
        max_to_upper_gap = upper - max(embd_one[i],embd_two[i])
        min_to_lower_gap = min(embd_one[i],embd_two[i]) - lower
        if step<max_to_upper_gap:
            val = max(embd_one[i],embd_two[i])+step
        elif step<min_to_lower_gap:
            val = min(embd_one[i],embd_two[i])-step
        else:
            if max_to_upper_gap>min_to_lower_gap:
                val = rng.uniform(upper-(0.8*max_to_upper_gap),upper)
            else:
                val = rng.uniform(lower,lower+(0.2*max_to_upper_gap))
        embd.append(val)
    return np.array(embd)

def get_random_embedding_with_condition(rng: np.random, d: int, lower: float, upper: float,
                                        embeddings: list, previous_inds: Tuple[int,int], n_tries_max=10000) -> np.array:
    """
    Generate random embedding r_3 such that rel(r_1,r_2) > rel(r_1,r_3) and rel(r_1,r_2) > rel(r_2,r_3)
    
    See original trans2vec article for reference on anchor embeddings

    Args:
        rng: random generator
        d: number of dimensions
        lower: lower bound
        upper: upper bound
        embeddings: list of embeddings
        previous_inds: indexes of r_1 and r_2 in embeddings
        
    Returns:
        New embedding
    """
    prev_rel = compute_embeddings_pairwise_relevance(embeddings[previous_inds[0]], embeddings[previous_inds[1]])
    #print(f'get_random_embedding_with_condition : prev_rel = {prev_rel}')
    embd = []
    cpt_tries = 0
    is_ok = False
    while cpt_tries<n_tries_max and not is_ok:
        embd = get_random_embedding(rng, d, lower, upper)
        #print(f'rngen => {embd}')
        is_ok = meets_conditions(prev_rel, embeddings[previous_inds[0]], embeddings[previous_inds[1]], embd)
        cpt_tries += 1
    # if no satisfying solution is found by random generation
    if cpt_tries>=n_tries_max and not is_ok:
        #print('(adjusting manually...)')
        embd = adjust_embd_manually(rng, embeddings[previous_inds[0]], embeddings[previous_inds[1]], lower, upper)
    #print(f'=> embedding = {embd}')
    rel_one_three = compute_embeddings_pairwise_relevance(embeddings[previous_inds[0]], embd)
    rel_two_three = compute_embeddings_pairwise_relevance(embeddings[previous_inds[1]], embd)
    #c = ((prev_rel > rel_one_three) and (prev_rel > rel_two_three))
    #print(f'meets_conditions : {c} | {prev_rel} | {rel_one_three} | {rel_two_three}')
    return embd

def init_anchor_embeddings_from_sim_mat(sim_mat: list, d: int, seed=12) -> list:
    """
    Generate anchor embeddings based on the input relevance/similarity matrix
    
    See original trans2vec article for reference on anchor embeddings

    Args:
        sim_mat: similarity matrix
        d: number of dimensions
        
    Returns:
        List of anchor embeddings
    """
    pair_to_rel = {}
    n = len(sim_mat)
    for i in range(n):
        for j in range(i+1,n):
            key = (i,j)
            pair_to_rel[key] = sim_mat[i][j]
    #ordered_pair_to_rel = dict(sorted(pair_to_rel.items(), key=lambda item: item[1]))
    ordered_pairs = [k for k, _ in sorted(pair_to_rel.items(), key=lambda item: item[1], reverse=True)]
    #print(ordered_pairs[:20])
    embeddings = [np.zeros(d) for _ in range(n)]
    defined = [False for _ in range(n)]
    lower_bound = -1./(2.*float(d))
    upper_bound = 1./(2.*float(d))
    rng = np.random.default_rng(seed)
    init_embd_one = get_random_embedding(rng, d, lower_bound, upper_bound)
    init_embd_two = get_random_embedding(rng, d, lower_bound, upper_bound)
    #print(init_embd_one)
    #print(init_embd_two)
    embeddings[ordered_pairs[0][0]] = init_embd_one
    defined[ordered_pairs[0][0]] = True
    embeddings[ordered_pairs[0][1]] = init_embd_two
    defined[ordered_pairs[0][1]] = True
    for i in range(1,n):
        previous_inds = ordered_pairs[i-1]
        ind_one = ordered_pairs[i][0]
        ind_two = ordered_pairs[i][1]
        #print(f'Iteration {i} : {previous_inds} ; {ind_one}, {ind_two}')
        if not defined[ind_one]:
            embeddings[ind_one] = get_random_embedding_with_condition(rng, d, lower_bound, upper_bound,
                                                                      embeddings, previous_inds)
            defined[ind_one] = True
        if not defined[ind_two]:
            embeddings[ind_two] = get_random_embedding_with_condition(rng, d, lower_bound, upper_bound,
                                                                      embeddings, previous_inds)
            defined[ind_two] = True
    return embeddings


def normalize_feat(d: dict, feat: str) -> dict:
    """
    Normalize numeric feature to a [0,1] range
    
    Args:
        d: input data
        feat: feature to normalize
        
    Returns:
        Data with feat normalized
    """
    vals = [d[t][feat] for t in d if d[t][feat]>0]
    #print(vals)
    new_d = {}
    vals_min = min(vals)
    vals_max = max(vals)
    for t in d:
        new_val = (d[t][feat] - vals_min) / (vals_max - vals_min)
        entry = d[t].copy()
        entry[feat] = new_val
        new_d[t] = entry
    return new_d

def infer_features_type(d: dict) -> dict:
    """
    Find out if features are numeric or categoric
    
    Args:
        d: input data
        
    Returns:
        Type for each feature
    """
    ignore = ['name', 'link']
    first_key = next(iter(d))
    feats = list(d[first_key])
    feat_to_type = {}
    for feat in feats:
        if feat not in ignore:
            f_type = 'categoric'
            if isinstance(d[first_key][feat], float):
                f_type = 'numeric'
            feat_to_type[feat] = f_type
    return feat_to_type


def compute_sim_mat(tourneys_dict: dict, feats_to_normalize: list) -> Tuple[list, list]:
    """
    Compute similarity matrix from tournaments features file
    
    Args:
        tourneys_dict: tournaments features as dictionary
        feats_to_normalize: numeric features to normalize
        
    Returns:
        Similarity matrix, list of tournaments names
    """
    tourneys = list(tourneys_dict.keys())
    n = len(tourneys_dict)
    sim_mat = [ [0.]*n for _ in range(n)]
    for i in range(n):
        sim_mat[i][i] = 1
    d = copy.deepcopy(tourneys_dict)
    for ftn in feats_to_normalize:
        d = normalize_feat(d, ftn)
    feats_types = infer_features_type(d)
    for i in range(n):
        for j in range(i+1,n):
            diff_sum = 0.
            for feat in feats_types:
                val_i = d[tourneys[i]][feat]
                val_j = d[tourneys[j]][feat]
                diff_val = 1.
                if feats_types[feat]=='numeric':
                    diff_val = abs(val_i-val_j)
                else:
                    if val_i==val_j:
                        diff_val = 0.
                diff_sum += diff_val
            diff = 1.-(round(diff_sum/float(len(feats_types)), 3))
            sim_mat[i][j] = diff
            sim_mat[j][i] = diff
    return sim_mat, tourneys


def read_feats_as_dict(infile: str, i_index=0, ignore=[]) -> dict:
    """
    Store data from csv as dict
    
    Not very pretty, but enables to reuse the compute_sim_mat function already implemented and tested
    
    Args:
        infile: input csv file
        i_index: column index of feature that should be used as key
        ignore: list of features to ignore
        
    Returns:
        Data dict with one entry per input row
    """
    feats_dict = {}
    df = pd.read_csv(infile)
    feats_names = df.columns.values.tolist()
    feat_index = feats_names[i_index]
    to_ignore = [feat_index]+ignore
    for _, row in df.iterrows():
        entry = {feat: row[feat] for feat in feats_names if feat!=to_ignore}
        feats_dict[row[feat_index]] = entry
    return feats_dict

def compute_anchor_embedding_contexts(tourneys_feats_file, d):
    """
    v1, deprecated
    """
    print('Running compute_anchor_embedding_contexts...')
    tourneys_feats = read_feats_as_dict(tourneys_feats_file, ignore=['name','link'])
    sim_mat, tourneys = compute_sim_mat(tourneys_feats, feats_to_normalize=['last_prize_pool'])
    anchor_embds = init_anchor_embeddings_from_sim_mat(sim_mat, d)
    #print(f'sample embedding 0 = {anchor_embds[0]}')
    #print(f'sample embedding {len(anchor_embds)-1} = {anchor_embds[-1]}')
    #for i in range(20):
    #    print(f'Embedding {i} : {anchor_embds[i]}')
    return anchor_embds


def map_column(x, d):
    if x in d:
        return d[x]
    #print(f'warning: no entry for key {x}')
    return -1

def is_in_dict(x, d):
    return (x in d)


def format_players_dataframe(players_feats_file: str, players_dict: dict) -> pd.DataFrame:
    """
    One-hot encoding for categorical features
    
    Args:
        players_feats_file: input players features file
        players dict: str->int players mapping
        
    Returns:
        Preprocessed players info dataframe
    """
    df = pd.read_csv(players_feats_file)
    to_drop = ['country','name']
    categorical_feats = ['hand','backhand_type','subregion']
    for cf in categorical_feats:
        unique_vals = df[cf].unique().tolist()
        n_u = len(unique_vals)
        val_to_ind = {unique_vals[i]: i for i in range(n_u)}
        values = df[cf].to_list()
        n = len(values)
        new_feats_vals = [[0]*n for _ in range(n_u)]
        for i in range(n):
            new_feats_vals[val_to_ind[values[i]]][i] = 1
        for j in range(n_u):
            c_val_name = f'{cf}_{str(unique_vals[j]).replace(" ", "-")}'
            df[c_val_name] = new_feats_vals[j]
    df.drop(categorical_feats, axis=1, inplace=True)
    df['name'] = df['name'].apply(map_column, args=[players_dict])
    df = df[df['name']!=-1].copy()
    print(f'=> df = {df.shape}')
    #df['has_results'] = df['name'].apply(is_in_dict, args=[player_to_results])
    #print(df[df['has_results']==False])
    #df = df[df['has_results']==True].copy()
    #print(f'=> df = {df.shape}')
    df.drop(to_drop, axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f'=> df = {df.shape}')
    return df

def load_players_data(players_feats_file: str, edges_pc: list, players_dict: dict) -> Tuple[pd.DataFrame,pd.DataFrame]:
    """
    Read and preprocess players features, and store in two dataframes
    
    The first one has labels depending on the match result expressed in each edge (1/-1 for win/loss),
    rows are duplicated to represent each win and loss, i.e. there is one row per match result per player
    The second one only has players features, i.e. one row per player
    
    Args:
        players_feats_file: input players features file
        edges_pc: list of edges, edge = [player,context,wins,losses]
        players dict: str->int players mapping
        
    Returns:
        Dataframe with labels, dataframe with players features only
    """
    player_to_results = {}
    for edge in edges_pc:
        p = edge[0]
        wins = edge[2]
        losses = edge[3]
        entry = {'wins': 0, 'losses': 0}
        if p in player_to_results:
            entry = player_to_results[p]
        entry['wins'] += wins
        entry['losses'] += losses
        player_to_results[p] = entry
    df_raw = format_players_dataframe(players_feats_file, players_dict)
    #print(df_raw.head())
    player_to_data = df_raw.T.to_dict('list')
    print(f'{len(player_to_data)}')
    data_lists = []
    for player in player_to_data.keys():
        results = player_to_results[player]
        for _ in range(results['wins']):
            data = player_to_data[player].copy()
            data.append(1)
            data_lists.append(data)
        for _ in range(results['losses']):
            data = player_to_data[player].copy()
            data.append(0)
            data_lists.append(data)
    #data_np = np.asarray(data_lists)
    df_with_labels = pd.DataFrame(data=data_lists)
    print(f'df_with_labels : {df_with_labels.shape} | df_raw : {df_raw.shape}')
    return df_with_labels, df_raw

    
def load_contexts_raw_data(contexts_feats_file: str) -> pd.DataFrame:
    """
    Similar to 'load_players_data', but only building features dataframe
    
    It does not make sense for a context to have a label,
    since there is always one winner and one loser in each match, so it would cancel out
    (see high level project documentation for more details)
    
    Args:
        players_feats_file: input players features file
        edges_pc: list of edges, edge = [player,context,wins,losses]
        players dict: str->int players mapping
        
    Returns:
        Dataframe with labels, dataframe with players features only
    """
    df = pd.read_csv(contexts_feats_file)
    to_drop = ['name_stage','name','link','location_country']
    categorical_feats = ['surface','type','subregion','round_tier']
    for cf in categorical_feats:
        unique_vals = df[cf].unique().tolist()
        n_u = len(unique_vals)
        val_to_ind = {unique_vals[i]: i for i in range(n_u)}
        values = df[cf].to_list()
        n = len(values)
        new_feats_vals = [[0]*n for _ in range(n_u)]
        for i in range(n):
            new_feats_vals[val_to_ind[values[i]]][i] = 1
        for j in range(n_u):
            c_val_name = f'{cf}_{str(unique_vals[j]).replace(" ", "-")}'
            df[c_val_name] = new_feats_vals[j]
    df.drop(categorical_feats, axis=1, inplace=True)
    df.drop(to_drop, axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f'=> df_contexts_raw = {df.shape}')
    return df


# we can only have binary (= categorical encoded) or numeric features
def infer_df_features_type(df: pd.DataFrame) -> list:
    """
    Same as 'infer_features_type', applied to a dataframe instead of a dict
    
    Args:
        df: input DataFrame
        
    Returns:
        List of features types
    """
    types = []
    feats = df.columns.values.tolist()
    for ifeat in range(len(feats)):
        unique_vals = df[feats[ifeat]].unique().tolist()
        if len(unique_vals)==2:
            types.append('binary')
        else:
            types.append('numeric')
    return types


def i_func(x, y):
    if x==y:
        return 1
    return 0

def compute_players_relevance(edges: list, w: list, users_feats: pd.DataFrame)-> dict:
    """
    Compute closest neighbors (from edges) players pairwise relevance
    
    Args:
        edges: list of closest neighbors
        w: features weights
        users_feats: DataFrame with players features
        
    Returns:
        Dict with closest neighbors players relevance
    """
    #X, _ = split_xy(users_feats)
    X = users_feats.values
    rels_u = {}
    for edge in edges:
        u = edge[0]
        up = edge[1]
        u_feats = X[u]
        up_feats = X[up]
        rel = 0.
        for i in range(len(w)):
            rel += w[i]*i_func(u_feats[i],up_feats[i])
        rel /= sum(w)
        new_dict = {}
        if u in rels_u.keys():
            new_dict = rels_u[u]
        new_dict[up] = rel
        rels_u[u] = new_dict
    return rels_u


def i_func_t(x, y, t):
    if t=='binary':
        if x==y:
            return 1
        return 0
    return abs(x-y)


def compute_relevance(edges: list, feats: pd.DataFrame, weights: list) -> dict:
    """
    Cleaner/more general version of 'compute_players_relevance'
    
    Can be used for both players and contexts
    Supports case where no weights are used
    Used in v2
    
    Args:
        edges: list of closest neighbors
        feats: DataFrame with players features
        weights: features weights
        
    Returns:
        Dict with closest neighbors players relevance
    """
    if len(weights)==0:
        weights = [1. for _ in range(feats.shape[1])]
    types = infer_df_features_type(feats)
    X = feats.values
    rels_u = {}
    for edge in edges:
        u = edge[0]
        up = edge[1]
        u_feats = X[u]
        up_feats = X[up]
        rel = 0.
        for i in range(len(weights)):
            rel += weights[i]*i_func_t(u_feats[i],up_feats[i], types[i])
        rel /= sum(weights)
        new_dict = {}
        if u in rels_u.keys():
            new_dict = rels_u[u]
        new_dict[up] = rel
        rels_u[u] = new_dict
    return rels_u

def split_xy(data: pd.DataFrame) -> Tuple[pd.DataFrame,pd.DataFrame]:
    y = data.iloc[:,-1]
    X = data.copy()
    X.drop(X.columns[-1], axis=1, inplace=True)
    return X, y

def get_edges_from_kdtree(data: pd.DataFrame, k_nn: int, weights: list) -> list:
    """
    Find closest neighbors from data
    
    Args:
        data: features
        k_nn: number of closest neighbors to return
        weights: features weights, if any
        
    Returns:
        List of [int,int] edges describing closest neighbor relationships between elements
    """
    edges = []
    #X, _ = split_xy(data)
    X = data.values
    if len(weights)>0:
        X = data.values * weights
    tree = KDTree(X)
    _, ind_to_k_neighbors = tree.query(X, k=k_nn+1)       # need to add 1 since the point itself is returned as part of the kNNs
    for i in range(len(ind_to_k_neighbors)):
        knns = ind_to_k_neighbors[i]
        for j in range(1,len(knns)):
            edge = [knns[0],knns[j]]
            edges.append(edge)
    return edges

def learn_weights(data):
    X, y = split_xy(data)
    #print(X)
    reg = LinearRegression().fit(X, y)
    #df_coeffs = pd.DataFrame(zip(X.columns, reg.coef_))
    coeffs_np = np.asarray(reg.coef_)
    return coeffs_np

def fix_opposed_embeddings(d: int) -> Tuple[np.array,np.array]:
    """
    Fix embeddings that represent the two match outcomes: win or loss
    
    Args:
        d: number of dimensions
        
    Returns:
        'loss' embedding, 'win' embedding
    """
    lower_bound = -1./(2.*float(d))
    upper_bound = 1./(2.*float(d))
    x_0 = [lower_bound for _ in range(d)]
    x_1 = [upper_bound for _ in range(d)]
    return [np.array(x_0),np.array(x_1)]

def get_random_other(elts, i_used):
    possibles = []
    for i in range(len(elts)):
        if i!=i_used:
            possibles.append(elts[i])
    rnd = random.choice(possibles)
    return rnd

def has_converged(epsilon, vects_before, vects_after):
    """
    Compare vectors before and after, check if they have changed by less than epsilon in total
    
    Args:
        epsilon: convergence threshold
        vects_before: list of vectors before iteration
        vects_after: list of vectors after iteration
        
    Returns:
        Dict with closest neighbors players relevance
    """
    if len(vects_before)!=len(vects_after):
        print('warning: found mismatched vector length when computing convergence (STOPPING)')
        return True
    sum_norm = 0
    for i in range(len(vects_before)):
        xone_np = np.asarray(vects_after[i])
        xone_p_np = np.asarray(vects_before[i])
        norm_one = np.linalg.norm(xone_np-xone_p_np)
        sum_norm += norm_one
    conv = ((sum_norm)<=epsilon)
    print(f'{sum_norm}')
    return conv

# Player2Vec
def learn_players_embeddings(edges_pc, users_data, users_data_raw, x_c, n_players, n_dims,
                             alpha, beta_one, beta_two, epsilon, k_nn):
    """
    v1, only learning players embeddings (cf. doc)
    """
    print('Init embeddings...')
    x_p = init_embeddings(n_players, n_dims)
    print('Learning weights...')
    w_ps = learn_weights(users_data)
    print('Getting edges from kdtree...')
    edges_pp = get_edges_from_kdtree(users_data_raw, w_ps, k_nn)
    print('Computing user relevance...')
    rels_p = compute_players_relevance(edges_pp, w_ps, users_data_raw)
    #modes_inds = [i for i in range(len(x_m))]
    contexts_inds = [i for i in range(len(x_c))]
    iters = 1
    converged = False
    while not converged and iters<10:
        print('iteration '+str(iters))
        x_p_prev = x_p.copy()
        for edge_pp in edges_pp:
            #print(edge_pp)
            p = edge_pp[0]
            pb = edge_pp[1]
            xp = x_p[p]
            xpb = x_p[pb]
            x_p[p] = xp - (alpha*beta_one/iters)*(np.dot(xp,xpb) - rels_p[p][pb])*xpb
            x_p[pb] = xpb - (alpha*beta_one/iters)*(np.dot(xp,xpb) - rels_p[p][pb])*xp
        for edge_pc in edges_pc:
            #print(edge_pc)
            p = edge_pc[0]
            c = edge_pc[1]
            n_wins = edge_pc[2]
            n_losses = edge_pc[3]
            #xp = x_p[p]
            xc = x_c[c]
            for _ in range(n_wins):
                xp = x_p[p]
                cb = get_random_other(contexts_inds, c)
                xcb = x_c[cb]
                x_p[p] = xp - (alpha/iters)*(sigmoid(np.dot(xp,xc)) - 1)*xc - (alpha/iters)*(sigmoid(np.dot(xp,xcb)))*xcb
            for _ in range(n_losses):
                xp = x_p[p]
                cb = get_random_other(contexts_inds, c)
                xcb = x_c[cb]
                x_p[p] = xp + (alpha/iters)*(sigmoid(np.dot(xp,xc)) - 1)*xc + (alpha/iters)*(sigmoid(np.dot(xp,xcb)))*xcb
        converged = has_converged(epsilon, [x_p_prev], [x_p])
        iters += 1
    return x_p


def learn_joint_embeddings(edges_pc, players_data, players_data_raw, contexts_data_raw, n_dims,
                           alpha, beta_one, beta_two, epsilon, k_nn):
    """
    v2, joint learning of players and contexts embeddings
    """
    print('Init embeddings...')
    x_p = init_embeddings(players_data_raw.shape[0], n_dims)
    x_c = init_embeddings(contexts_data_raw.shape[0], n_dims)
    x_r = fix_opposed_embeddings(n_dims)        # [loss_embd,win_embd]
    print('Learning weights...')
    w_ps = learn_weights(players_data)
    print('Getting edges from kdtree...')
    edges_pp = get_edges_from_kdtree(players_data_raw, k_nn, weights=w_ps)
    edges_cc = get_edges_from_kdtree(contexts_data_raw, k_nn, weights=[])
    print('Computing user relevance...')
    rels_p = compute_relevance(edges_pp, players_data_raw, weights=w_ps)
    rels_c = compute_relevance(edges_cc, contexts_data_raw, weights=[])
    #modes_inds = [i for i in range(len(x_m))]
    contexts_inds = [i for i in range(len(x_c))]
    iters = 1
    converged = False
    while not converged and iters<10:
        print('iteration '+str(iters))
        x_p_prev = x_p.copy()
        x_c_prev = x_c.copy()
        for edge_pp in edges_pp:
            #print(edge_pp)
            p = edge_pp[0]
            pb = edge_pp[1]
            xp = x_p[p]
            xpb = x_p[pb]
            x_p[p] = xp - (alpha*beta_one/iters)*(np.dot(xp,xpb) - rels_p[p][pb])*xpb
            x_p[pb] = xpb - (alpha*beta_one/iters)*(np.dot(xp,xpb) - rels_p[p][pb])*xp
        for edge_cc in edges_cc:
            #print(edge_pp)
            c = edge_cc[0]
            cb = edge_cc[1]
            xc = x_c[c]
            xcb = x_c[cb]
            x_c[c] = xc - (alpha*beta_one/iters)*(np.dot(xc,xcb) - rels_c[c][cb])*xcb
            x_c[cb] = xcb - (alpha*beta_one/iters)*(np.dot(xc,xcb) - rels_c[c][cb])*xc
        for edge_pc in edges_pc:
            #print(edge_pc)
            p = edge_pc[0]
            n_wins = edge_pc[2]
            n_losses = edge_pc[3]
            #xp = x_p[p]
            for _ in range(n_wins):
                xp = x_p[p]
                xr = x_r[1]
                #xrb = x_r[0]
                #x_p[p] = xp - (alpha/iters)*(sigmoid(np.dot(xp,xr)) - 1)*xr - (alpha/iters)*(sigmoid(np.dot(xp,xrb)))*xrb
                x_p[p] = xp - (alpha/iters)*(sigmoid(np.dot(xp,xr)) - 1)*xr
            for _ in range(n_losses):
                xp = x_p[p]
                xr = x_r[0]
                #xrb = x_r[1]
                #x_p[p] = xp - (alpha/iters)*(sigmoid(np.dot(xp,xr)) - 1)*xr - (alpha/iters)*(sigmoid(np.dot(xp,xrb)))*xrb
                x_p[p] = xp - (alpha/iters)*(sigmoid(np.dot(xp,xr)) - 1)*xr
        converged = has_converged(epsilon, [x_p_prev,x_c_prev], [x_p,x_c])
        iters += 1
    return x_p, x_c, x_r


def compute_outcome(x_p, x_c, p_one, p_two, c):
    xc = x_c[c]
    xp_one = x_p[p_one]
    xp_two = x_p[p_two]
    score_one = np.dot(xp_one,xc)
    score_two = np.dot(xp_two,xc)
    return score_one, score_two


def evaluate_embeddings_on_test_folder(folder_path, x_p, x_c, players_dict, contexts_dict):
    match_results = load_results(folder_path, players_dict, contexts_dict)
    right = 0
    wrong = 0
    for res in match_results:
        score_one, score_two = compute_outcome(x_p, x_c, res[0], res[1], res[2])
        if score_one>score_two:
            pred_outcome = 1
        else:
            pred_outcome = 2
        real_outcome = res[3]
        if pred_outcome==real_outcome:
            right += 1
        else:
            wrong += 1
    print(f'right = {right} | wrong = {wrong}')

    
def compute_outcome_v2(x_p, x_c, x_r, p_one, p_two, c, gamma):
    xc = x_c[c]
    xp_one = x_p[p_one]
    xp_two = x_p[p_two]
    xr = x_r[1]
    score_one = gamma*(np.dot(xp_one,xr)) + (1.-gamma)*(np.dot(xc,xr))
    score_two = gamma*(np.dot(xp_two,xr)) + (1.-gamma)*(np.dot(xc,xr))
    return score_one, score_two

def evaluate_embeddings_on_test_folder_v2(folder_path, x_p, x_c, x_r, players_dict, contexts_dict, gamma):
    match_results = load_results(folder_path, players_dict, contexts_dict)
    right = 0
    wrong = 0
    for res in match_results:
        score_one, score_two = compute_outcome_v2(x_p, x_c, x_r, res[0], res[1], res[2], gamma)
        if score_one>score_two:
            pred_outcome = 1
        else:
            pred_outcome = 2
        real_outcome = res[3]
        if pred_outcome==real_outcome:
            right += 1
        else:
            wrong += 1
    print(f'right = {right} | wrong = {wrong}')


def p2vec_wrapper(players_feats_file, tourneys_feats_file, results_folder, test_folder):
    """
    v1
    """
    n_dims = 32     # 64
    alpha = 0.4
    beta_one = 0.1
    beta_two = 0.3
    epsilon = 0.01  # ??
    k_nn = 5
    #players_list, players_dict = load_list_and_map(players_feats_file, col='name')
    contexts_list, contexts_dict = load_list_and_map(tourneys_feats_file, col='name_stage')
    x_c = compute_anchor_embedding_contexts(tourneys_feats_file, n_dims)
    #x_r = compute_anchor_embedding_results(tourneys_feats_file)
    edges_pc, players_list, players_dict = load_edges_and_players_from_results(players_feats_file,
                                                                               results_folder, contexts_dict)
    players_data, players_data_raw = load_players_data(players_feats_file, edges_pc, players_dict)
    n_players = players_data_raw.shape[0]
    x_p = learn_players_embeddings(edges_pc, players_data, players_data_raw, x_c,
                                   n_players, n_dims, alpha, beta_one, beta_two, epsilon, k_nn)
    evaluate_embeddings_on_test_folder(test_folder, x_p, x_c, players_dict, contexts_dict)


def p2vec_v2_wrapper(players_feats_file, tourneys_feats_file, results_folder, test_folder):
    """
    v2
    """
    n_dims = 32     # 64
    alpha = 0.4
    beta_one = 0.1
    beta_two = 0.3
    epsilon = 0.01  # ??
    k_nn = 5
    gamma = 0.5
    contexts_list, contexts_dict = load_list_and_map(tourneys_feats_file, col='name_stage')
    #edges_cc = compute_contexts_intra_edges(tourneys_feats_file, n_dims)
    edges_pc, players_list, players_dict = load_edges_and_players_from_results(players_feats_file,
                                                                               results_folder, contexts_dict)
    players_data, players_data_raw = load_players_data(players_feats_file, edges_pc, players_dict)
    contexts_data_raw = load_contexts_raw_data(tourneys_feats_file)
    x_p, x_c, x_r = learn_joint_embeddings(edges_pc, players_data, players_data_raw, contexts_data_raw,
                                   n_dims, alpha, beta_one, beta_two, epsilon, k_nn)
    evaluate_embeddings_on_test_folder_v2(test_folder, x_p, x_c, x_r, players_dict, contexts_dict, gamma)

def p2vec_v2_wrapper():
    """
    v2 using config file
    """
    n_dims = cfg.N_DIMS
    alpha = cfg.ALPHA
    beta_one = cfg.BETA_ONE
    beta_two = cfg.BETA_TWO
    epsilon = cfg.EPSILON
    k_nn = cfg.K_NN
    gamma = cfg.GAMMA
    players_feats_file = os.path.join(cfg_dp.prepped_data_path, 'inp_players.csv')
    tourneys_feats_file = os.path.join(cfg_dp.prepped_data_path, 'inp_tourneys.csv')
    results_folder = cfg.train_data_folder
    test_folder = cfg.test_data_folder
    contexts_list, contexts_dict = load_list_and_map(tourneys_feats_file, col='name_stage')
    edges_pc, players_list, players_dict = load_edges_and_players_from_results(players_feats_file,
                                                                               results_folder, contexts_dict)
    players_data, players_data_raw = load_players_data(players_feats_file, edges_pc, players_dict)
    contexts_data_raw = load_contexts_raw_data(tourneys_feats_file)
    x_p, x_c, x_r = learn_joint_embeddings(edges_pc, players_data, players_data_raw, contexts_data_raw,
                                   n_dims, alpha, beta_one, beta_two, epsilon, k_nn)
    evaluate_embeddings_on_test_folder_v2(test_folder, x_p, x_c, x_r, players_dict, contexts_dict, gamma)


if __name__=='__main__':
    players_file = 'E:/projects/gsm/data/model_input/inp_players.csv'
    tourneys_file = 'E:/projects/gsm/data/model_input/inp_tourneys.csv'
    results_folder = 'E:/projects/gsm/data/results'
    test_folder = 'E:/projects/gsm/data/results_test'
    #p2vec_wrapper(players_file, tourneys_file, results_folder, test_folder)
    #p2vec_v2_wrapper(players_file, tourneys_file, results_folder, test_folder)
    p2vec_v2_wrapper()
