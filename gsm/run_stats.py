import os
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import gsm.conf.conf_data_prep as cfgdp
import gsm.conf.conf_webscrapping as cfgws
import gsm.data.data_prep as dp

"""
Make graphs of stats for players, tournaments, and results
"""

def make_hist(labels: list, vals: list, out_file: str, title: str):
    fig, ax = plt.subplots()
    ax.bar(labels, vals)
    plt.suptitle(title)
    plt.savefig(out_file)
    plt.close(fig)

def make_hist_from_list(elts: list, out_file: str, title: str):
    cnt_tmp = Counter(elts)
    cnt = {}
    for key in cnt_tmp:
        if key>0:
            cnt[key] = cnt_tmp[key]
    min_key = min(cnt.keys())
    max_key = max(cnt.keys())
    labels = []
    vals = []
    for i in range(min_key,max_key+1):
        if i in cnt:
            vals.append(cnt[i])
        else:
            vals.append(0)
        labels.append(i)
    #plt.bar(labels, vals)
    make_hist(labels, vals, out_file, title)

def make_pie_chart_from_list(elts: list, out_file: str, title: str):
    cnt = Counter(elts)
    labels = []
    vals = []
    for key in cnt:
        labels.append(key)
        vals.append(cnt[key])
    fig, ax = plt.subplots()
    ax.pie(vals, labels=labels, autopct='%1.1f%%')
    plt.suptitle(title)
    plt.savefig(out_file)
    plt.close(fig)


def make_stats_players(players_file: str, out_folder: str):
    df = pd.read_csv(players_file)
    df['yob'] = df['dob'].apply(dp.get_yob_as_int)
    yobs = df['yob'].to_list()
    hands = df['hand'].to_list()
    bhtype = df['backhand_type'].to_list()
    make_hist_from_list(yobs, os.path.join(out_folder, 'graph_players_yob.png'), '#players by year of birth')
    make_pie_chart_from_list(hands, os.path.join(out_folder, 'graph_players_hand.png'), 'Main hand')
    make_pie_chart_from_list(bhtype, os.path.join(out_folder, 'graph_players_bhtype.png'), 'Backhand type')

def make_stats_tourneys(tourneys_prep_file: str, out_folder: str):
    df = pd.read_csv(tourneys_prep_file)
    df = df[df['round_tier']==0].copy()
    types = df['type'].to_list()
    make_pie_chart_from_list(types, os.path.join(out_folder, 'graph_tourneys_type.png'), 'Tournaments by type')

def make_stats_results(results_folder: str, out_folder: str):
    dirs = [d for d in os.listdir(results_folder) if os.path.isdir(os.path.join(results_folder, d))]
    labels = []
    vals = []
    for d in dirs:
        label = int(d)
        n = 0
        subfolder = os.path.join(results_folder, d)
        files = [d for d in os.listdir(subfolder) if os.path.isfile(os.path.join(subfolder, d))]
        for f in files:
            f_path = os.path.join(subfolder, f)
            with open(f_path, 'r') as f:
                lines = f.readlines()
                n_lines = len(lines)-2
                if len(lines[-1])<3:
                    n_lines -= 1
                n += n_lines
        if label>=1968:
            labels.append(label)
            vals.append(n)
    make_hist(labels, vals, os.path.join(out_folder, 'graph_results_year.png'), '#match results by year')

if __name__=='__main__':
    players_file = os.path.join(cfgws.players_data_path, 'players.csv')
    tourneys_prep_file = os.path.join(cfgdp.prepped_data_path, 'inp_tourneys.csv')
    make_stats_players(players_file, cfgdp.graphs_stats_path)
    make_stats_tourneys(tourneys_prep_file, cfgdp.graphs_stats_path)
    make_stats_results(cfgws.results_data_path, cfgdp.graphs_stats_path)