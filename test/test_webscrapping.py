import unittest
import numpy as np
import sys
import os
#sys.path.append(os.path.abspath(os.path.join('..', 'gsm')))
from gsm.data.webscrapping import WikiScrapper

class WikiScrapperUnitTests(unittest.TestCase):
    '''Unit tests for the WikiScrapper class'''
    
    def test_parse_root_page(self):
        print("Testing parsing the root ATP page on Wikipedia")
        #pass
        tourney_types = ['Grand Slam','ATP Masters 1000','ATP 500','ATP 250']
        scrapper = WikiScrapper()
        parsed_tourney_types = scrapper.parse_root_page()
        #print(f'parsed_tourney_types : {parsed_tourney_types}')
        n_matches = 0
        for ptt in parsed_tourney_types:
            if ptt['name'] in tourney_types:
                n_matches += 1
        self.assertEqual(n_matches, len(tourney_types))
        
    def test_parse_level_type(self):
        print("Testing parsing the ATP Masters 1000 page on Wikipedia")
        #pass
        tourney_names = ['Indian Wells Masters','Miami Open','Monte-Carlo Masters','Madrid Open',
                         'Italian Open','Canadian Open','Cincinnati Masters','Shanghai Masters','Paris Masters']
        scrapper = WikiScrapper()
        ttype = {'name': 'ATP Masters 1000', 'link': '/wiki/ATP_Tour_Masters_1000'}
        parsed_tourneys = scrapper.parse_level_type(ttype)
        n_matches = 0
        for pt in parsed_tourneys:
            if pt['name'] in tourney_names:
                n_matches += 1
        self.assertEqual(n_matches, len(tourney_names))
        
    def test_parse_player_info(self):
        print("Testing parsing a player page on Wikipedia")
        #pass
        player_info = {'name': 'Dominic Thiem', 'height': '1.85', 'country': 'Austria', 'dob': '1993-09-03',
                       'hand': 'right', 'backhand_type': 'one'}
        scrapper = WikiScrapper()
        rplink = '/wiki/Dominic_Thiem'
        parsed_player_info = scrapper.parse_player_info(rplink)
        n_matches = 0
        for field in parsed_player_info:
            if field in player_info:
                if parsed_player_info[field]==player_info[field]:
                    n_matches += 1
        self.assertEqual(n_matches, len(player_info))


#unittest.main()