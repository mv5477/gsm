import os
import data.webscrapping as ws
import data.data_prep as dp
import re

def build_input_files(results_path, players_path, countries_file, prepped_data_path):
    tourneys_prepped_file = os.path.join(prepped_data_path, 'inp_tourneys.csv')
    dp.build_tourneys_file(results_path, countries_file, tourneys_prepped_file)
    players_file = os.path.join(players_path, 'players.csv')
    players_prepped_file = os.path.join(prepped_data_path, 'inp_players.csv')
    #dp.build_players_file(players_file, countries_file, players_prepped_file)
    #results_prepped_file = os.path.join(prepped_data_path, 'inp_results.csv')
    #dp.build_results_file(results_path, results_prepped_file)


if __name__=='__main__':
    scrapper = ws.WikiScrapper()
    results_path = scrapper.get_results_path()
    players_path = scrapper.get_players_path()
    #prepped_data_path = 'E:/projects/gsm/data/model_input/'
    #countries_file = 'E:/projects/gsm/data/utils/countries.csv'
    build_input_files(results_path, players_path, countries_file, prepped_data_path)
    '''s = 'US$60,102,000'
    si = s.replace(',', '')
    si = re.sub('\D', ' ', si).strip().split(' ')[0]
    print(f'{si}')'''