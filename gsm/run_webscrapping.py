import gsm.data.webscrapping as ws

"""
Simple wrapper calling the parsing of web pages to find results and players
"""

if __name__=='__main__':
    scrapper = ws.WikiScrapper()
    #scrapper = ws.ATPScrapper()
    #scrapper.parse_from_root()
    #scrapper.parse_full_tree()
    #scrapper.test_current_step()
    #scrapper.test_incremental()
    scrapper.parse_full_tree_incremental(exclude_tourney_types=['ATP Finals'])
