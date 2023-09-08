import os
import gsm.data.webscrapping as ws
import gsm.data.data_prep as dp


def build_input_files(results_path: str, players_path: str, countries_file: str, prepped_data_path: str):
    """
    Calls data_prep functions to process results folder and raw players info
    
    Args:
        results_path: path to the folder with raw match results (output of webscrapping)
        players_path: path to the folder with raw player info (output of webscrapping)
        countries_file: path to the helper file mapping countries to subregions
        prepped_data_path: path to the output folder
    """
    tourneys_prepped_file = os.path.join(prepped_data_path, 'inp_tourneys.csv')
    dp.build_tourneys_file(results_path, countries_file, tourneys_prepped_file)
    players_file = os.path.join(players_path, 'players.csv')
    players_prepped_file = os.path.join(prepped_data_path, 'inp_players.csv')
    dp.build_players_file(players_file, countries_file, players_prepped_file)


if __name__=='__main__':
    scrapper = ws.WikiScrapper()
    results_path = scrapper.get_results_path()
    players_path = scrapper.get_players_path()
    build_input_files(results_path, players_path, countries_file, prepped_data_path)