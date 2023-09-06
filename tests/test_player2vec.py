import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..', 'gsm')))
import gsm.models.player2vec as p2vec
import gsm.conf.conf_player2vec as cfg

class Player2VecUnitTests(unittest.TestCase):
    '''Unit tests for the player2vec module'''
    
    def test_infer_winner_from_scorelines(self):
        print("Checking process to infer winner from players scorelines")
        with self.subTest():
            scoreline = '[[6, 6], [2, 4]]'
            self.assertEqual(p2vec.infer_winner_from_scoreline(scoreline), 1)
        with self.subTest():
            scoreline = '[[6, 6, 5], [2, 7, 7]]'
            self.assertEqual(p2vec.infer_winner_from_scoreline(scoreline), 2)
        with self.subTest():
            scoreline = '[[1], []]'
            self.assertEqual(p2vec.infer_winner_from_scoreline(scoreline), 1)

    def test_random_embedding_with_condition(self):
        print("Checking process to generate random anchor embeddings")
        rng = np.random.default_rng(1)
        d = 4
        lower_bound = -1./(2.*float(d))
        upper_bound = 1./(2.*float(d))
        embeddings = [np.array([0.11,0.09,0.05,0.13]), np.array([-0.01,-0.03,0.02,-0.06])]
        embd = p2vec.get_random_embedding_with_condition(rng, d, lower_bound, upper_bound, embeddings, [0,1])
        prev_rel = p2vec.compute_embeddings_pairwise_relevance(embeddings[0], embeddings[1])
        rel_one_three = p2vec.compute_embeddings_pairwise_relevance(embeddings[0], embd)
        rel_two_three = p2vec.compute_embeddings_pairwise_relevance(embeddings[1], embd)
        c = ((prev_rel > rel_one_three) and (prev_rel > rel_two_three))
        self.assertIs(c, True)


unittest.main()