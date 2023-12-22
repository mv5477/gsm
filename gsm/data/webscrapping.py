import os
import json
import re
import copy
from typing import Tuple
import requests
import pandas as pd
from bs4 import BeautifulSoup
#sys.path.append(os.path.abspath(os.path.join('..', 'gsm')))
import gsm.conf.conf_webscrapping as cfg

class WikiScrapper:
    """
    Functions to scrap and parse data from Wikipedia
    """
    wikipedia_page = ''
    root_page = ''

    def __init__(self):
        self.wikipedia_page = 'https://en.wikipedia.org'
        self.root_page = '/wiki/ATP_Tour'
        self.data_folder_results = cfg.results_data_path
        self.data_folder_players = cfg.players_data_path

    def get_results_path(self) -> str:
        return self.data_folder_results

    def get_players_path(self) -> str:
        return self.data_folder_players


    def parse_root_page_sidebar(self):
        """
        DEPRECATED
        """
        page = requests.get(self.root_page)
        #with open(rp_code) as rp:
        soup = BeautifulSoup(page.text, 'html.parser')
        #print('got the soup')
        #tourney_types_list = soup.find('a', string='ATP Tour')
        dds = soup.find('dd')
        #print(dds)
        entries = []
        for dd in dds.children:
            items = dd.find_all('a')
            for item in items:
                #print(item.get_text())
                #print(item['href'])
                entry = {'name': item.get_text(), 'link': item['href']}
                entries.append(entry)
        return entries


    def parse_root_page(self)-> list:
        """
        Parse tournaments types from the root (ATP) page
        
        Returns:
            List of tournaments types (name and link), e.g. 'ATP 500'
        """
        page = requests.get(self.wikipedia_page+self.root_page)
        soup = BeautifulSoup(page.text, 'html.parser')
        table = soup.find_all('table')[1]
        ntourneys_total = 0
        entries = []
        trs = table.find_all('tr')
        #print(str(len(trs)))
        for itr in range(1,len(trs)):
            all_as = trs[itr].find_all('a')
            if all_as:
                info = all_as[-1]
                t_name = info.get_text()
                t_link = info['href']
                entry = {'name': t_name, 'link': t_link}
                entries.append(entry)
                t_ntourneys = int(trs[itr].find('td').get_text())
                ntourneys_total += t_ntourneys
            else:
                break
        print(f'Found {ntourneys_total} tournament types (from root page)')
        return entries

    def extract_first_int(self, s: str) -> int:
        """
        Parse first integer in string
        
        Args:
            s: input string
            
        Returns:
            Integer value
        """
        if isinstance(s, int):
            return int(s)
        si = re.sub('\D', ' ', s).strip().split(' ')[0]
        if si:
            return int(si)
        return -1

    def extract_prize_money(self, s: str) -> int:
        """
        Similar to extract_first_int, handling prize moneys formats
        
        Args:
            s: input string
            
        Returns:
            Integer value
        """
        if isinstance(s, int):
            return int(s)
        si = s.replace(',', '')
        si = re.sub('\D', ' ', si).strip().split(' ')[0]
        if si:
            if 'million' in s:
                si *= 1000000
            return int(si)
        return -1

    def format_surface(self, raw_surf: str) -> str:
        """
        Formatting and removing brackets for hard courts (Grand Slam table mostly)
        
        Args:
            raw_surf: input string
            
        Returns:
            'Hard' or 'Hard (indoor)'
        """
        surf = raw_surf
        if 'hard' in raw_surf.lower():
            if 'indoor' in raw_surf.lower():
                surf = 'Hard (indoor)'
            else:
                surf = 'Hard'
        return surf

    def parse_tourneys_table(self, table, ttype_name: str) -> list:
        """
        Parse html table containing the list of tournaments
        
        Args:
            table: BeautifulSoup object with parsed HTML table
            ttype_name: tournament type
            
        Returns:
            List of entries (dicts) with tournaments info
        """
        i_surface = i_drawsize = 0
        entries = []
        trs = table.find_all('tr')
        for tr in trs:
            if tr.find('td') and tr.find('th'):
                #print(tr)
                potential_infos = tr.find('th').find_all('a')
                info = ''
                for pi in potential_infos:
                    if pi.get_text():
                        info = pi
                        break
                t_name = info.get_text()
                t_link = info['href']
                tds = tr.find_all(['th','td'])
                t_surface = self.format_surface(tds[i_surface].get_text().strip())
                t_drawsize = self.extract_first_int(tds[i_drawsize].get_text())
                if ttype_name=='Grand Slam':
                    t_drawsize = 128
                entry = {'name': t_name, 'link': t_link, 'surface': t_surface, 'drawsize': t_drawsize}
                #print(entry)
                entries.append(entry)
            elif tr.find('th', string=re.compile('^Surface')):
                ths = tr.find_all(['th','td'])
                for ith in range(len(ths)):
                    txt = ths[ith].get_text()
                    if txt.startswith('Surface'):
                        i_surface = ith
                    elif txt.startswith('Draw'):
                        i_drawsize = ith
                #print(f'{i_surface} | {i_drawsize}')
        #print(str(len(entries)))
        return entries

    def parse_level_type(self, ttype: dict) -> list:
        """
        Parse tournament type page (e.g. the ATP 500 page)
        
        Args:
            ttype: input type (name and link)
            
        Returns:
            List of entries (dicts) with tournaments info
        """
        page = requests.get(self.wikipedia_page+ttype['link'])
        soup = BeautifulSoup(page.text, 'html.parser')
        tourney_list = []
        if soup.find_all(id='Tournaments'):
            #print('found tournaments list')
            tables = soup.find_all('table')
            table_to_parse = ''
            for table in tables:
                if table.find('th', string=re.compile('^Surface')):
                    table_to_parse = table
                    break
            tourney_list = self.parse_tourneys_table(table_to_parse, ttype['name'])
        else:
            #print('no list')
            entry = {'name': ttype['name'], 'link': ttype['link'], 'surface': 'unknown', 'drawsize': -1}
            tourney_list.append(entry)
        #print(str(len(tourney_list)))
        return tourney_list


    def find_and_parse_tourney_infobox(self, page_soup, tourney_info: dict) -> dict:
        """
        Parse infobox on tournament page, adding prize pool and location country
        
        Args:
            page_soup: BeautifulSoup object of tournament page
            tourney_info: current tournament info
            
        Returns:
            Updated tournament info
        """
        #infobox = page_soup.find('table', class_='infobox vevent')
        infobox = page_soup.find('table', class_=re.compile('^infobox'))
        new_tourney_info = copy.deepcopy(tourney_info)
        location_country = ''
        last_prize_pool = -1
        trs = infobox.find_all('tr')
        for tr in trs:
            if tr.find('th', string='Location'):
                td = tr.find('td')
                toks = td.get_text('|').split('|')
                country = toks[-1]
                if '(' in country or '[' in country:
                    found = False
                    i = -2
                    while not found and -i<=len(toks):
                        if len(toks[i].strip())>2:
                            found = True
                            break
                        i -= 1
                    country = toks[i].strip()
                location_country = country
            elif tr.find('th', string=re.compile('^Prize')):
                td = tr.find('td')
                i_prize = self.extract_prize_money(td.get_text())
                last_prize_pool = i_prize
        new_tourney_info['location_country'] = location_country
        new_tourney_info['last_prize_pool'] = last_prize_pool
        return new_tourney_info

    def parse_level_tourney(self, tourney: dict) -> Tuple[list, dict]:
        """
        Parse tournament page to gather the list of editions, and get additional info from infobox
        
        Args:
            tourney: input tournament (name and link)
            
        Returns:
            List of editions (= years), updated tournament info
        """
        open_era_cutoff = 1969
        page = requests.get(self.wikipedia_page+tourney['link'])
        soup = BeautifulSoup(page.text, 'html.parser')
        new_tourney_info = self.find_and_parse_tourney_infobox(soup, tourney)
        editions_list = []
        #items = soup.find('div', class_='navbox-styles')        # first occurence
        items = soup.find_all('div', style='padding:0 0.25em')[0]
        editions = items.find_all('a', string=re.compile('[0-9]+'))
        last_ed = 0
        for ied in range(len(editions)-1,0,-1):
            ed = editions[ied]
            if ed.has_attr('href'):
                ed_year = self.extract_first_int(ed.get_text())
                last_ed = ed_year
                break
        print(f'Last edition = {last_ed}')
        if last_ed>0 and last_ed<open_era_cutoff:
            items = soup.find_all('div', style='padding:0 0.25em')[1]
            editions = items.find_all('a', string=re.compile('[0-9]+'))
        for ed in editions:
            if ed.has_attr('href'):
                ed_year = self.extract_first_int(ed.get_text())
                ed_link = ed['href']
                if 'redlink' not in ed_link:
                    ed_entry = copy.deepcopy(tourney)
                    ed_entry['link'] = ed_link
                    ed_entry['edition'] = ed_year
                    editions_list.append(ed_entry)
        tname = tourney['name']
        print(f'{len(editions_list)} editions for tournament {tname}')
        return editions_list, new_tourney_info


    def parse_level_edition(self, ted: dict) -> dict:
        """
        Parse tournament edition page to find link to (mens singles) results page
        
        Args:
            ted: input edition (name and link)
            
        Returns:
            Edition info with link to results page
        """
        print(ted['link'])
        page = requests.get(self.wikipedia_page+ted['link'])
        soup = BeautifulSoup(page.text, 'html.parser')
        tsp = {}
        is_wta = False
        is_atp = False
        nvts = soup.find_all('li', class_='nv-talk')
        for nvt in nvts:
            firsta = nvt.find('a')
            #print(firsta)
            if firsta.has_attr('href'):
                if 'WTA' in firsta['href']:
                    is_wta = True
                if 'ATP' in firsta['href']:
                    is_atp = True
        #print(f'is_wta = {is_wta}')
        items = soup.find_all('div', role='note')
        if items and is_atp:
            iitem = -1
            for i in range(len(items)):
                raw_text = items[i].get_text().lower()
                #if raw_text.endswith('singles') and not 'women' in raw_text:
                if 'singles' in raw_text and 'women' not in raw_text and 'qualif' not in raw_text:
                    iitem = i
                    break
            if iitem>-1:
                mens_singles = items[iitem]
                tsp = copy.deepcopy(ted)
                tsp['link'] = mens_singles.find('a')['href']
        print(tsp)
        return tsp


    def infer_rounds_labels(self, tr_row) -> list:
        """
        Harmonize rounds labels
        
        Args:
            tr_row: BeautifulSoup object of results table
            
        Returns:
            List of rounds labels
        """
        tds = tr_row.find_all('td')
        raw_labels = [td.get_text().lower().strip() for td in tds if len(td.get_text())>3]
        labels_mapping = {'main_draw': {'first round': '1R', 'second round': '2R',
                          'third round': '3R', 'fourth round': '4R',
                          'quarterfinals': 'QF', 'quarter-finals': 'QF',
                          'semifinals': 'SF', 'semi-finals': 'SF',
                          'finals': 'F', 'final': 'F'},
                          'qualifiers': {'first round': 'Q1R', 'second round': 'Q2R', 'qualifying competition': 'QQ',
                          'first preliminary round': 'Q1R', 'second preliminary round': 'Q2R'}}
        clean_labels = []
        draw_type = 'main_draw'
        if 'qualifying competition' in raw_labels:
            draw_type = 'qualifiers'
        for raw_label in raw_labels:
            if raw_label in labels_mapping[draw_type]:
                clean_labels.append(labels_mapping[draw_type][raw_label])
            else:
                clean_labels.append(raw_label)
        return clean_labels

    def fix_scoreline(self, scoreline: list) -> list:
        """
        Catch edge cases in set results
        
        Args:
            scoreline: input string
            
        Returns:
            Cleaned scoreline
        """
        new_sl = []
        for set_score in scoreline:
            iss = set_score
            if len(str(set_score))>1:
                iss = int(str(set_score)[0])
            new_sl.append(iss)
        return new_sl

    def extract_players_scorelines(self, tr_rows) -> list:
        """
        Parse scorelines from the tree result in Wikipedia pages
        
        Args:
            page_soup: BeautifulSoup object of results tree
            
        Returns:
            List of scores for each line
        """
        #regex_align_center = re.compile('^text-align:center')
        sls_per_row = []
        for tr_row in tr_rows:
            sls_row = []
            current_player = ''
            current_scoreline = []
            match_is_over = False
            tds = tr_row.find_all('td')
            for td_row in tds:
                row_as = td_row.find_all('a', title=re.compile('.+'))
                #score_content = re.match(regex_align_center, td_row['style'])
                score_content = ''
                if td_row.has_attr('style'):
                    if current_player and not match_is_over:
                        cell_content = td_row.get_text().strip()
                        if cell_content=='':
                            match_is_over = True
                        else:
                            if td_row['style'].startswith('text-align:center'):
                                score_content = td_row.get_text().strip()
                #eol = tr_row.find('td', style=re.compile('^border'))
                if row_as:
                    if current_player:
                        sls_row.append({'player': current_player, 'scoreline': self.fix_scoreline(current_scoreline)})
                        current_player = ''
                        current_scoreline = []
                        match_is_over = False
                    player = row_as[-1]
                    current_player = player['href']
                    #print(f'new player : {current_player}')
                elif score_content:
                    score_set = self.extract_first_int(score_content)
                    if score_set>-1:
                        current_scoreline.append(score_set)
                        #print(f'new score : {score_set}')
                    elif 'w/o' in score_content:
                        current_scoreline.append(1)
                        #print(f'new score : walk-over => 1')
            if current_player:
                sls_row.append({'player': current_player, 'scoreline': self.fix_scoreline(current_scoreline)})
            if sls_row:
                sls_per_row.append(sls_row)
        return sls_per_row


    def scorelines_to_matches(self, scorelines_rows: list, rounds_labels: list) -> list:
        """
        Reconstruct match results from rows scorelines
        
        Args:
            scorelines_rows: list of scores for each row
            rounds_labels: rounds labels
            
        Returns:
            List of match results
        """
        mappings = [[[1],[1,2],[1,2],[1,3],[1,3],[1,2],[1,2],[1,4],[1,4],[1,2],[1,2],[1,3],[1,3],[1,2],[1,2],[1]],          # Section table, 4 levels, no first round byes
                    [[2],[1,2],[1,3],[1,3],[1,2],[2],[4],[4],[2],[1,2],[1,3],[1,3],[1,2],[2]],                              # Section table, 4 levels, byes for top 8 seeds
                    [[2],[1,2],[1,3],[1,3],[1,2],[1,2],[1,4],[1,4],[1,2],[1,2],[1,3],[1,3],[1,2],[2]],
                    [[2],[1,2],[1,3],[1,3],[1,2],[1,2],[1,4],[4],[2],[1,2],[1,3],[1,3],[1,2],[1,2],[1]],                      # Section table, 4 levels, byes for top 4 seeds (top)
                    [[1],[1,2],[1,2],[1,3],[1,3],[1,2],[2],[4],[1,4],[1,2],[1,2],[1,3],[1,3],[1,2],[2]],                      # Section table, 4 levels, byes for top 4 seeds (bottom)
                    [[2],[1,2],[1,3],[1,3],[1,2],[2],[4],[4],[2],[1,2],[1,3],[1,3],[2],[1,2],[1]],                          # Section table, 4 levels, byes for top 4 seeds (again)
                    [[1],[1,2],[2],[3],[3],[2],[1,2],[1,4],[4],[2],[1,2],[1,3],[3],[2],[1,2],[1]],                          # Section table, 4 levels, byes for top 4 seeds (and again)
                    [[2],[1,2],[1,3],[1,3],[1,2],[2],[4],[1,4],[1,2],[2],[3],[1,3],[1,2],[2]],                              # Section table, 4 levels, byes for top 4 seeds (and again)
                    [[1],[1],[2],[2],[1],[1],[3],[3],[1],[1],[2],[2],[1],[1],[4],[4],[1],[1],[2],[2],[1],
                    [1],[3],[3],[1],[1],[2],[2],[1],[1]],
                    [[1],[1],[2],[2],[1],[1],[3],[3],[1],[1],[2],[2],[1],[1]],                                              # Finals table, QF to F
                    [[1],[1],[2],[2],[1],[1]],                                                                              # Finals table, SF to F
                    [[1],[1,2],[1,2],[1,3],[1,3],[1,2],[1,2],[1,4],[1,4],[1,2],[1,2],[1,3],[1,3],[1,2],[1,2],[1,5],[1,5],   # One-table five-round (e.g. Rotterdam 2002)
                    [1,2],[1,2],[1,3],[1,3],[1,2],[1,2],[1,4],[1,4],[1,2],[1,2],[1,3],[1,3],[1,2],[1,2],[0]],
                    [[1],[1,2],[1,2],[1]],
                    [[3],[3],[4],[1,4],[1,2],[1,2],[1,3],[3],[5],[5],[3],[3],[4],[1,4],[1,2],[1,2],[1,3],[3]],              # 2 rounds of qualifiers + 3 rounds main draw (e.g. French Open 1972)
                    [[2],[1,2],[1,3],[3],[2],[1,2],[1,4],[1,4],[1,2],[2],[3],[1,3],[1,2],[2]],                              # Vienna Open 1979
                    [[1],[1,2],[2],[3],[1,3],[1,2],[2],[4],[4],[2],[1,2],[1,3],[3],[2],[1,2],[1]],
                    [[1],[1,2],[2],[3],[1,3],[1,2],[2],[4],[1,4],[1,2],[2],[3],[3],[2],[1,2],[1]],
                    [[2],[1,2],[1,3],[1,3],[1,2],[2]],                                                                      # Qualifiers
                    [[2],[2],[1],[1]],
                    [[2],[1,2],[1,3],[1,3],[1,2],[1,2],[1]],
                    [[1],[1,2],[1,2],[1,3],[1,3],[1,2],[1,2],[1]],                                                          # Qualifiers / generic 3-rounder
                    [[1],[1]],                                                                                              # Finals only
                    ]
        mapping_to_use = []
        matches = []
        for mp in mappings:
            if len(mp)==len(scorelines_rows):
                is_ok = True
                for im in range(len(scorelines_rows)):
                    if len(scorelines_rows[im])!=len(mp[im]):
                        is_ok = False
                if is_ok:
                    mapping_to_use = mp
                    break
        if not mapping_to_use:
            lens = []
            for im in range(len(scorelines_rows)):
                lens.append(len(scorelines_rows[im]))
            print('warning: could not find matching mapping for scores :')
            #print(scorelines_rows)
            print(lens)
            return matches
        current_matches = {}
        for im in range(len(scorelines_rows)):
            scorelines = scorelines_rows[im]
            mapping = mapping_to_use[im]
            for ielt in range(len(scorelines)):
                match_round = mapping[ielt]
                sl = scorelines[ielt]
                if match_round in current_matches:
                    match_details = current_matches[match_round]
                    match_details['player_two'] = sl['player']
                    match_details['score'].append(sl['scoreline'])
                    matches.append(match_details)
                    del current_matches[match_round]
                else:
                    match_round_label = rounds_labels[match_round-1]
                    match_details = {'player_one': sl['player'], 'score': [sl['scoreline']], 'round': match_round_label}
                    current_matches[match_round] = match_details
        return matches


    def parse_level_results(self, tsp: dict) -> list:
        """
        Parse edition results page
        
        Args:
            tsp: input results page (name and link)
            
        Returns:
            List of match results
        """
        print(tsp['link'])
        page = requests.get(self.wikipedia_page+tsp['link'])
        soup = BeautifulSoup(page.text, 'html.parser')
        '''draw_size = 1
        n_seeds = 1
        info_box = soup.find('table', class_='infobox')
        info_trs = info_box.find_all('tr')
        for tr in info_trs:
            if tr.find('th', class_='infobox-label', string='Draw'):
                s_draw_size = tr.find('td').get_text()
                draw_size = self.extract_first_int(s_draw_size)
            if tr.find('th', class_='infobox-label', string='Seeds'):
                s_n_seeds = tr.find('td').get_text()
                n_seeds = self.extract_first_int(s_n_seeds)
        has_first_round_byes = ((draw_size%n_seeds)>0)'''
        all_match_results = []
        tables = soup.find_all('table', cellpadding='0')
        for table in tables:
            tr_rows = table.find_all('tr')
            rounds_labels = self.infer_rounds_labels(tr_rows[0])
            players_sls = self.extract_players_scorelines(tr_rows[1:])
            matches_ress = self.scorelines_to_matches(players_sls, rounds_labels)
            all_match_results += matches_ress
        return all_match_results


    def is_valid_link(self, rplink: str) -> bool:
        return (str(rplink).startswith('/wiki'))

    def parse_height(self, s: str) -> str:
        #h_regex = re.compile(r'\d\.\d\d(.*)(m)')
        h_regex = re.compile(r'\d\.\d\d')
        result = h_regex.search(s)
        if result:
            return result.group()
        return '-1'

    def parse_hands(self, s: str) -> Tuple[str, str]:
        """
        Parse handedness and backhand type
        
        Args:
            s: input string
            
        Returns:
            Handedness (left/right), backhand type (one/two)
        """
        hand = 'unknown'
        backhand_type = 'unknown'
        lows = s.lower()
        if 'right' in lows:
            hand = 'right'
        elif 'left' in lows:
            hand = 'left'
        if 'one-hand' in lows:
            backhand_type = 'one'
        elif 'two-hand' in lows:
            backhand_type = 'two'
        return hand, backhand_type

    def parse_dob(self, s: str) -> str:
        h_regex = re.compile(r'\d\d\d\d-\d\d-\d\d')
        result = h_regex.search(s)
        if result:
            return result.group()
        return 'unknown'

    def parse_player_info(self, rplink: str) -> dict:
        """
        Parse player page
        
        Args:
            rplink: page link
            
        Returns:
            Player info (name, country, etc.)
        """
        player_info = {'name': 'unknown', 'height': -1, 'country': 'unknown', 'dob': 'unknown',
                       'hand': 'unknown', 'backhand_type': 'unknown'}
        #print(rplink)
        link_is_valid = self.is_valid_link(rplink)
        if link_is_valid:
            p_name = rplink.split('/')[-1].replace('_', ' ').replace(' (tennis)', '')
            player_info['name'] = p_name
            page = requests.get(self.wikipedia_page+rplink)
            soup = BeautifulSoup(page.text, 'html.parser')
            #infobox = soup.find('table', class_='infobox vcard')
            infobox = soup.find('table', class_=re.compile('infobox(.+)vcard'))
            if infobox:
                trs = infobox.find_all('tr')
                for tr in trs:
                    th = tr.find('th')
                    td = tr.find('td')
                    if th and td:
                        header = th.get_text()
                        #print(td.get_text())
                        if header.startswith('Country'):
                            toks = td.get_text('|').split('|')
                            country = toks[-1]
                            if '(' in country or '[' in country:
                                found = False
                                i = -2
                                while not found and -i<=len(toks):
                                    if len(toks[i].strip())>2:
                                        found = True
                                        break
                                    i -= 1
                                country = toks[i].strip()
                            player_info['country'] = country
                        elif header.startswith('Born'):
                            dob = self.parse_dob(td.get_text())
                            player_info['dob'] = dob
                        elif header.startswith('Height'):
                            #height = td.get_text().split(' ')[0]
                            height = self.parse_height(td.get_text())
                            player_info['height'] = height
                        elif header.startswith('Plays'):
                            hand, backhand_type = self.parse_hands(td.get_text())
                            player_info['hand'] = hand
                            player_info['backhand_type'] = backhand_type
        else:
            p_name = rplink.split('title=')[1].split('&')[0].replace('_', ' ').replace(' (tennis)', '')
            player_info['name'] = p_name
        return player_info

    def parse_players_from_match_results(self, all_match_results: list) -> dict:
        """
        DEPRECATED
        """
        print(f'Parsing players from {len(all_match_results)} match results')
        raw_players_links = {}
        for m_res in all_match_results:
            p_one = m_res['player_one']
            p_two = m_res['player_two']
            if p_one not in raw_players_links:
                raw_players_links[p_one] = 1
            if p_two not in raw_players_links:
                raw_players_links[p_two] = 1
        print(f'Found {len(raw_players_links)} unique players')
        players_dict = {}
        p_id = 0
        for rplink in raw_players_links:
            player_info = self.parse_player_info(rplink)
            print(player_info)
            players_dict[p_id] = player_info
            p_id += 1
        return players_dict

    def parse_players_if_new(self, all_match_results: list, all_players_dict: dict) -> dict:
        """
        Parse player info if not already known
        
        Args:
            all_match_results: list of match results to extract players from
            all_players_dict: dict of already known players
            
        Returns:
            Dict of new players (key = name)
        """
        print(f'Parsing players from {len(all_match_results)} match results')
        raw_players_links = {}
        for m_res in all_match_results:
            p_one = m_res['player_one']
            p_two = m_res['player_two']
            if (p_one not in all_players_dict) and (p_one not in raw_players_links):
                raw_players_links[p_one] = 1
            if (p_two not in all_players_dict) and (p_two not in raw_players_links):
                raw_players_links[p_two] = 1
        print(f'Found {len(raw_players_links)} new unique players')
        players_dict = {}
        for rplink in raw_players_links:
            player_info = self.parse_player_info(rplink)
            print(player_info)
            if ' ' in player_info['name']:        # catch (some) weird cases (country names instead of players)
                players_dict[rplink] = player_info
        return players_dict

    def update_players_dict_from_match_results(self, players_dict: dict, match_results: list) -> dict:
        """
        Add new players info
        
        Args:
            players_dict: dict of already known players
            match_results: list of match results to extract players from
            
        Returns:
            Updated dict of players info
        """
        new_dict = copy.deepcopy(players_dict)
        app_dict = self.parse_players_if_new(match_results, players_dict)
        for app_key in app_dict:
            new_dict[app_key] = app_dict[app_key]
        return new_dict

    def make_dataframe_results(self, m_ress: list) -> pd.DataFrame:
        """
        Convert list of match results to dataframe
        
        Args:
            m_ress: match results
            
        Returns:
            Dataframe of match results
        """
        data_rows = []
        feats = ['player_one','player_two','round','score']
        for m_res in m_ress:
            data_row = []
            for feat in feats:
                data_row.append(m_res[feat])
            data_rows.append(data_row)
        df = pd.DataFrame(data=data_rows, columns=feats)
        df.drop_duplicates(subset=['player_one', 'player_two', 'round'], inplace=True)
        return df

    def parse_full_tree(self):
        """
        DEPRECATED
        """
        tourney_types = self.parse_root_page()
        #self.parse_level_type('https://en.wikipedia.org/wiki/ATP_250')
        tourneys = []
        for t_type in tourney_types:
            new_tourneys = self.parse_level_type(t_type)
            tourneys += new_tourneys
        print(f'Found {len(tourneys)} tournaments')
        t_editions = []
        for tourney in tourneys:
            new_t_editions = self.parse_level_tourney(tourney)
            if len(new_t_editions)>1:
                t_editions += new_t_editions
        print(f'Found {len(t_editions)} tournament editions')
        t_singles_pages = []
        for ted in t_editions:
            tsp = self.parse_level_edition(ted)
            if tsp:
                t_singles_pages.append(tsp)
        print(f'Found {len(t_singles_pages)} corresponding singles pages')


    def get_dict_from_csv(self, csvfile: str, key_field: str) -> dict:
        """
        Read csv data and build dict of players info
        
        Args:
            csvfile: players csv file
            key_field: dataframe column to use as dict key
            
        Returns:
            Dict of players info
        """
        out_dict = {}
        df = pd.read_csv(csvfile)
        other_fields = df.columns.values.tolist()
        ind_kf = -1
        for ifield in range(len(other_fields)):
            if other_fields[ifield]==key_field:
                ind_kf = ifield
                break
        other_fields.pop(ind_kf)
        for _, row in df.iterrows():
            elt = {field: row[field] for field in other_fields}
            out_dict[row[key_field]] = elt
        return out_dict


    def check_folder_exists(self, folder_path: str):
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            os.makedirs(folder_path)

    # key = wiki link
    def write_players_file(self, players_dict: dict, players_file: str):
        """
        (over)write players info file
        
        Args:
            players_dict: players info
            players_file: output file path
        """
        if players_dict:
            cols = list(players_dict[next(iter(players_dict))])
            data_rows = []
            for player_link in players_dict:
                data_row = [players_dict[player_link][col] for col in cols]
                data_row.append(player_link)
                data_rows.append(data_row)
            cols.append('wiki_link')
            df = pd.DataFrame(data=data_rows, columns=cols)
            df.to_csv(players_file, index=False)


    def parse_full_tree_incremental(self, overwrite=False, exclude_tourney_types=[]):
        """
        Full scrapping/parsing process, starting from the ATP tour root page
        
        Tournament results are organized by year: 1 folder per year, 1 file per tournament in this folder
        Players file is overwritten every time a results file is written
        If interrupted (manually or by an error), this process will restart where it stopped
        
        Args:
            overwrite: [NOT FULLY TESTED] restart the process from the beginning each time, overwriting everything
            exclude_tourney_types: types of tournaments to exclude from the parsing process
        """
        players_file = os.path.join(self.data_folder_players, 'players.csv')
        players_dict = {}
        if os.path.exists(players_file):
            players_dict = self.get_dict_from_csv(players_file, 'wiki_link')
        tourney_types = self.parse_root_page()
        for t_type in tourney_types:
            tourneys = self.parse_level_type(t_type)
            print(f'Found {len(tourneys)} tournaments for type {t_type}')
            for tourney in tourneys:
                t_editions, tourney_info = self.parse_level_tourney(tourney)
                tourney_info['type'] = t_type['name']
                if len(t_editions)>1 and tourney_info['type'] not in exclude_tourney_types:
                    print(f'Found {len(t_editions)} tournament editions')
                    for ted in t_editions:
                        s_year = str(ted['edition'])
                        s_name = tourney_info['name']
                        results_folder = os.path.join(self.data_folder_results, s_year)
                        results_file = os.path.join(results_folder, (s_year+'_'+s_name+'_results.csv'))
                        self.check_folder_exists(results_folder)
                        if not os.path.exists(results_file) or overwrite:
                            with open(results_file, 'w') as f:
                                f.write(json.dumps(tourney_info)+'\n')
                                f.close()
                            tsp = self.parse_level_edition(ted)
                            if tsp:
                                m_ress = self.parse_level_results(tsp)
                                players_dict = self.update_players_dict_from_match_results(players_dict, m_ress)
                                #self.write_results(tourney, ted, m_ress, results_file)
                                df_results = self.make_dataframe_results(m_ress)
                                df_results.to_csv(results_file, mode='a', index=False)
                                self.write_players_file(players_dict, players_file)


    def test_incremental(self, overwrite=False):
        """
        Test function
        """
        players_file = os.path.join(self.data_folder_players, 'players.csv')
        players_dict = {}
        if os.path.exists(players_file):
            players_dict = self.get_dict_from_csv(players_file, 'wiki_link')
        tourneys = []
        #mc = {'name': 'Monte-Carlo', 'link': '/wiki/Monte-Carlo_Masters'}
        #mc = {'name': 'Australian Open', 'link': '/wiki/Australian_Open'}
        mc = {'name': 'Marseille', 'link': '/wiki/Open_13'}
        #mc = {'name': 'Adelaide', 'link': '/wiki/Adelaide_International_(tennis)'}
        #mc = {'name': 'French Open', 'link': '/wiki/French_Open'}
        tourneys.append(mc)
        for tourney in tourneys:
            t_editions, tourney_info = self.parse_level_tourney(tourney)
            if len(t_editions)>1:
                print(f'Found {len(t_editions)} tournament editions')
                for ted in t_editions:
                    s_year = str(ted['edition'])
                    s_name = tourney_info['name']
                    results_folder = os.path.join(self.data_folder_results, s_year)
                    results_file = os.path.join(results_folder, (s_year+'_'+s_name+'_results.csv'))
                    self.check_folder_exists(results_folder)
                    if not os.path.exists(results_file) or overwrite:
                        with open(results_file, 'w') as f:
                            f.write(json.dumps(tourney_info)+'\n')
                            f.close()
                        tsp = self.parse_level_edition(ted)
                        if tsp:
                            m_ress = self.parse_level_results(tsp)
                            players_dict = self.update_players_dict_from_match_results(players_dict, m_ress)
                            df_results = self.make_dataframe_results(m_ress)
                            df_results.to_csv(results_file, mode='a', index=False)
                            self.write_players_file(players_dict, players_file)

    def test_current_step(self):
        """
        Test function
        """
        #ted = {'link': '/wiki/1972_Monte_Carlo_Open'}
        #tsp = self.parse_level_edition(ted)
        #print(tsp)
        tourneys = []
        #mc = {'name': 'Monte-Carlo', 'link': '/wiki/Monte-Carlo_Masters'}
        #mc = {'name': 'Australian Open', 'link': '/wiki/Australian_Open'}
        mc = {'name': 'Marseille', 'link': '/wiki/Open_13'}
        tourneys.append(mc)
        t_editions = []
        for tourney in tourneys:
            new_t_editions = self.parse_level_tourney(tourney)
            if len(new_t_editions)>1:
                t_editions += new_t_editions
        print(f'Found {len(t_editions)} tournament editions')
        t_singles_pages = []
        for ted in t_editions:
            tsp = self.parse_level_edition(ted)
            if tsp:
                t_singles_pages.append(tsp)
        print(f'Found {len(t_singles_pages)} corresponding singles pages')
        all_match_results = []
        for tsp in t_singles_pages:
            m_ress = self.parse_level_results(tsp)
            print(f'Found {len(m_ress)} matches results')
            #print(m_ress)
            all_match_results += m_ress
        print(f'Found {len(all_match_results)} matches')
        _ = self.parse_players_from_match_results(all_match_results)








class ATPScrapper:
    """
    Functions to scrap and parse data from the ATP website
    
    The results on this website are more complete, better formatted, and easier to parse than on Wikipedia
    Unfortunately, this site blocks web scrapping tools, so we could not use it
    Keeping this code just in case it becomes possible through some other way
    """
    atp_page = ''
    root_page = ''

    def __init__(self):
        self.atp_page = 'https://www.atptour.com'
        self.root_page = '/en/scores/results-archive'


    def parse_root_page(self):
        page = requests.get(self.atp_page+self.root_page)
        soup = BeautifulSoup(page.text, 'html.parser')
        print(soup)
        ul_years = soup.find('ul', id='resultsArchiveYearDropdown')
        print(ul_years)
        lis_years = ul_years.find_all('li')
        years_dict = {}
        for liy in lis_years:
            years_dict[liy['data-value']] = 1
        return list(years_dict.keys())


    def build_url_year(self, url_root, s_year):
        url_year = url_root+'?year='+str(s_year)
        return url_year


    def extract_info_from_res_page_url(self, url):
        toks = url.split('/')
        #s_year = toks[-2]
        s_t_id = toks[-3]
        return s_t_id


    def parse_level_year(self, url_year):
        page = requests.get(url_year)
        soup = BeautifulSoup(page.text, 'html.parser')
        print(soup)
        ul_tourneys = soup.find('ul', id='tournamentDropdown')
        lis_tourneys = ul_tourneys.find_all('li')
        tourneys_dict = {}
        for lit in lis_tourneys:
            tourneys_dict[lit['data-value']] = lit.get_text()
        tourneys_year = {}
        t_ress = soup.find_all('tr', class_='tourney-result')
        for t_res in t_ress:
            team = t_res.find_all('div', class_='tourney-detail-winner', string=re.compile('^Team'))
            if not team:
                res = t_res.find('a', string='Results')
                if res:
                    res_page_url = res.get_text()
                    entry = {'link': res_page_url}
                    t_id = extract_info_from_res_page_url(res_page_url)
                    tourneys_year[t_id] = entry
        return tourneys_dict, tourneys_year


    def update_tourneys_dict(self, t_dict_base, t_dict_app):
        for key in t_dict_app:
            if key not in t_dict_base:
                t_dict_base[key] = t_dict_app[key]
            #else:
                #do something?
        return t_dict_base


    def build_cutpoints_from_n_rounds(self, n_rounds):
        cutpoints = []
        sum_cpts = 0
        for i in n_rounds:
            n_matches_round = pow(2, i)
            sum_cpts += n_matches_round
            cutpoints.append(sum_cpts)
        return cutpoints


    def infer_round_from_index(self, index, cutpoints):
        for icpt in range(len(cutpoints)):
            if index<cutpoints[icpt]:
                return icpt
        print(f'warning: could not find round for index {index}')
        return -1


    def extract_player_id_from_url(self, url):
        toks = url.split('/')
        p_id = toks[-2]
        return p_id


    def parse_scoreline(self, s):
        sets = re.findall('[0-9][0-9]', s)
        p_one_score = []
        p_two_score = []
        for sset in sets:
            p_one_score.append(int(sset[0]))
            p_two_score.append(int(sset[1]))
        return (p_one_score, p_two_score)


    def parse_level_results(self, res_page_url, t_id, year):
        page = requests.get(self.atp_page+res_page_url)
        soup = BeautifulSoup(page.text, 'html.parser')
        div_ress = soup.find('div', id='scoresResultsContent')
        table_ress = div_ress.find('table', class_='day-table')
        n_rounds = len(table_ress.find_all('thead'))
        cpts = self.build_cutpoints_from_n_rounds(n_rounds)
        matches = table_ress.find_all('tbody')
        players = {}
        match_results = []
        i_match = 0
        for match in matches:
            players = match.find_all('td', class_='day-table-name')
            td_score = match.find('td', class_='day-table-score')
            if players[1].find('a'):
                p_one_url = players[0].find('a')['href']
                p_two_url = players[1].find('a')['href']
                p_one_id = self.extract_player_id_from_url(p_one_url)
                p_two_id = self.extract_player_id_from_url(p_two_url)
                players[p_one_id] = {'link': p_one_url}
                players[p_two_id] = {'link': p_two_url}
                score = self.parse_scoreline(td_score.get_text())
                i_round = self.infer_round_from_index(i_match, cpts)
                match_res = {'player_one': p_one_id, 'player_two': p_two_id, 'scoreline': score,
                             'tourney_id': t_id, 'year': year, 'round': i_round}
                match_results.append(match_res)
            i_match += 1
        return players, match_results


    def parse_full_tree(self):
        #years = self.parse_root_page()
        tourney_to_info = {}
        year_to_tourneys = {}
        players = {}
        match_results = []
        #for year in years:
        for year in ['2022']:
            url_year = self.build_url_year((self.atp_page+self.root_page), year)
            print(url_year)
            ts_dict, ts_year = self.parse_level_year(url_year)
            year_to_tourneys[year] = ts_year
            tourney_to_info = self.update_tourneys_dict(tourney_to_info, ts_dict)
            for t_id in ts_year:
                t_players, t_match_results = self.parse_level_results(t_id['link'], t_id, year)
                match_results += t_match_results
                players.update(t_players)

#if __name__=='__main__':
#    print('web scrapper')
