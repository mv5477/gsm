import sys
import os
import pandas as pd
import json
import copy
sys.path.append(os.path.abspath(os.path.join('..', 'gsm')))
import gsm.conf.conf_data_prep as cfg

def get_latest_tourney_info(results_folder):
    t_info_dict = {}
    dirs = [d for d in os.listdir(results_folder) if os.path.isdir(os.path.join(results_folder, d))]
    for d in dirs:
        #print(f'd = {d}')
        subfolder = os.path.join(results_folder, d)
        #print(f'subfolder = {subfolder}')
        files = [d for d in os.listdir(subfolder) if os.path.isfile(os.path.join(subfolder, d))]
        # assumes that files are read "in order" (same as writing order, i.e. chronological)
        for f in files:
            #print(f'f = {f}')
            f_path = os.path.join(subfolder, f)
            #print(f'f_path = {f_path}')
            f = open(f_path, 'r')
            header = f.readline()
            header_dict = json.loads(header)
            t_name = header_dict['name']
            t_info_dict[t_name] = header_dict
    return t_info_dict

def normalize_feat(d, feat):
    vals = [d[t][feat] for t in d if d[t][feat]>0]
    #print(vals)
    new_d = {}
    vals_min = min(vals)
    vals_max = max(vals)
    print(f'feature {feat} : min = {vals_min} | max = {vals_max}')
    for t in d:
        new_val = d[t][feat]
        if new_val>=0:
            new_val = (new_val-vals_min) / (vals_max-vals_min)
        entry = d[t].copy()
        entry[feat] = new_val
        new_d[t] = entry
    return new_d

def infer_features_type(d):
    #print(d)
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
        

def compute_sim_mat(tourneys_dict, feats_to_normalize):
    tourneys = list(tourneys_dict.keys())
    n = len(tourneys_dict)
    sim_mat = [ [0.]*n for _ in range(n)]
    for i in range(n):
        sim_mat[i][i] = 1
    d = copy.deepcopy(tourneys_dict)
    for ftn in feats_to_normalize:
        d = normalize_feat(d, ftn)
    for k in d:
        print(d[k])
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
    
    
def compute_features(tourneys_dict, feats_to_normalize):
    tourneys = list(tourneys_dict.keys())
    n = len(tourneys_dict)
    d = copy.deepcopy(tourneys_dict)
    for ftn in feats_to_normalize:
        d = normalize_feat(d, ftn)
    return d, tourneys
   

def get_country_to_region_dict(countries_file):
    df_countries = pd.read_csv(countries_file, sep=',', usecols=['name','sub-region'])
    country_to_region = dict(df_countries.values)
    return country_to_region

def write_sim_mat_to_csv(sim_mat, elts, outfile):
    data_rows = []
    for i in range(len(sim_mat)):
        data_row = [elts[i]]
        data_row += sim_mat[i]
        data_rows.append(data_row)
    df = pd.DataFrame(data_rows)
    df.columns = (['name']+elts)
    df.to_csv(outfile, index=False)
    
    
def write_tourneys_features_to_csv(feats_dict, elts, outfile, n_tiers=3):
    first_key = next(iter(feats_dict))
    feats_list = list(feats_dict[first_key])
    data_rows = []
    for elt in elts:
        for it in range(n_tiers):
            data_row = [f'{elt}_{it}']
            vals = feats_dict[elt]
            for feat in feats_list:
                val = 'unknown'
                if feat in vals and len(str(vals[feat]))>0:
                    val = vals[feat]
                data_row.append(val)
            data_row.append(it)
            data_rows.append(data_row)
    df = pd.DataFrame(data_rows)
    df.columns = (['name']+feats_list+['round_tier'])
    df.to_csv(outfile, index=False)
    
    
def write_players_features_to_csv(feats_dict, elts, outfile):
    first_key = next(iter(feats_dict))
    feats_list = list(feats_dict[first_key])
    data_rows = []
    for elt in elts:
        data_row = [elt]
        vals = feats_dict[elt]
        for feat in feats_list:
            val = 'unknown'
            if feat in vals and len(str(vals[feat]))>0:
                val = vals[feat]
            data_row.append(val)
        data_rows.append(data_row)
    df = pd.DataFrame(data_rows)
    df.columns = (['name']+feats_list)
    df.to_csv(outfile, index=False)


def build_tourneys_file_old(results_folder, tourneys_simmat_outfile):
    # read first line of each csv file to get tourney info
    # If discrepancy => ?? take most recent for now
    # Goal = square matrix with similarity between tourneys
    # => compute pairwise distance
    # features = type/rank, surface, indoors/outdoors, draw size, (most recent prize pool ?), continent/region
    print('Building tourneys relevance file...')
    country_to_region = get_country_to_region_dict(cfg.countries_file)
    tourneys_info = get_latest_tourney_info(results_folder)
    for t_name in tourneys_info:
        ti = tourneys_info[t_name]
        #print(ti)
        country = ti['location_country'].split(',')[-1].strip()
        subregion = ''
        if country in country_to_region:
            subregion = country_to_region[country]
        ti['subregion'] = subregion
    sim_mat, tourneys = compute_sim_mat(tourneys_info, feats_to_normalize=['last_prize_pool'])
    write_sim_mat_to_csv(sim_mat, tourneys, tourneys_simmat_outfile)
    
    
def build_tourneys_file(results_folder, tourneys_feats_outfile):
    '''
    Similarity only needs to be computed for a small neighborhood
    => compute/normalize tourney features here, and compute similarity at a later stage
    '''
    print('Building tourneys relevance file...')
    country_to_region = get_country_to_region_dict(cfg.countries_file)
    tourneys_info = get_latest_tourney_info(results_folder)
    for t_name in tourneys_info:
        ti = tourneys_info[t_name]
        #print(ti)
        #country = ti['location_country'].split(',')[-1].strip()
        country = ''
        subregion = ''
        country_toks = [tok.strip() for tok in ti['location_country'].split(',')]
        for c_tok in country_toks:
            if c_tok in country_to_region:
                country = c_tok
                subregion = country_to_region[country]
                break
        ti['location_country'] = country
        ti['subregion'] = subregion  
    t_feats, tourneys = compute_features(tourneys_info, feats_to_normalize=['last_prize_pool','drawsize'])
    write_tourneys_features_to_csv(t_feats, tourneys, tourneys_feats_outfile)
   

def get_yob_as_int(x):
    yob = -1
    s_yob = x.split('-')[0]
    #print(s_yob)
    #if isinstance(s_yob, int):
    if s_yob.isdigit():
        yob = int(s_yob)
    return yob

def read_csv_as_dict(infile, key_feat, attr_feats):
    df = pd.read_csv(infile)
    df['dob'] = df['dob'].apply(get_yob_as_int)
    out_dict = {}
    for _, row in df.iterrows():
        key = row[key_feat]
        entry = {f: row[f] for f in attr_feats}
        if not key in out_dict:                 # avoid duplicates (?)
            out_dict[key] = entry
    return out_dict

def build_players_file_old(players_file, players_simmat_outfile):
    '''
    Players "relevance" (= similarity) matrix
    '''
    print('Building players relevance file...')
    country_to_region = get_country_to_region_dict(cfg.countries_file)
    players_info = read_csv_as_dict(players_file, 'wiki_link', ['height','country','dob','hand','backhand_type'])
    for t_name in players_info:
        ti = players_info[t_name]
        #print(f'{t_name} : {ti}')
        subregion = ''
        if str(ti['country'])!='nan':
            country = ti['country'].split(',')[-1].strip()
            if country in country_to_region:
                subregion = country_to_region[country]
        ti['subregion'] = subregion
    sim_mat, players = compute_sim_mat(players_info, feats_to_normalize=['height','dob'])
    write_sim_mat_to_csv(sim_mat, players, players_simmat_outfile)
    
    
def build_players_file(players_file, players_feats_outfile):
    print('Building players relevance file...')
    country_to_region = get_country_to_region_dict(cfg.countries_file)
    players_info = read_csv_as_dict(players_file, 'wiki_link', ['height','country','dob','hand','backhand_type'])
    for t_name in players_info:
        ti = players_info[t_name]
        #print(f'{t_name} : {ti}')
        subregion = ''
        if str(ti['country'])!='nan':
            country = ti['country'].split(',')[-1].strip()
            ti['country'] = country
            if country in country_to_region:
                subregion = country_to_region[country]
        ti['subregion'] = subregion
    p_feats, players = compute_features(players_info, feats_to_normalize=['height','dob'])
    write_players_features_to_csv(p_feats, players, players_feats_outfile)
    
    
def build_results_file(results_path, results_prepped_file):
    df_out = pd.DataFrame()
    
    df_out.to_csv(results_prepped_file)