import unittest
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..', 'gsm')))
import gsm.data.data_prep as data_prep
import gsm.conf.conf_data_prep as cfg

class DataPrepUnitTests(unittest.TestCase):
    '''Unit tests for the data_prep module'''
    
    def test_countries_file(self):
        print("Cheking integrity of the countries reference file")
        df_countries = pd.read_csv(cfg.countries_file, sep=',', usecols=['name','sub-region'])
        subregions = df_countries['sub-region'].unique().tolist()
        self.assertEqual(len(subregions), 18)
        
    def test_normalize_feat(self):
        print("Cheking feature normalization process")
        feat = 'feat_to_norm'
        data_dict = {'id0': {feat: 20}, 'id1': {feat: 40}, 'id2': {feat: 70},
                     'id3': {feat: 200}, 'id4': {feat: 220}}
        normed = data_prep.normalize_feat(data_dict, feat)
        with self.subTest():
            np.testing.assert_almost_equal(normed['id0'][feat], 0.0)
        with self.subTest():
            np.testing.assert_almost_equal(normed['id1'][feat], 0.1)
        with self.subTest():
            np.testing.assert_almost_equal(normed['id2'][feat], 0.25)
        with self.subTest():
            np.testing.assert_almost_equal(normed['id3'][feat], 0.9)
        with self.subTest():
            np.testing.assert_almost_equal(normed['id4'][feat], 1.0)


unittest.main()