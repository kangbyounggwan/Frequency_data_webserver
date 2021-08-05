import numpy as np

from collections import defaultdict
from sklearn.utils import check_random_state

from rrcf import RCTree


MAX_INT = np.iinfo(np.int32).max


class RCForest(object):
    def __init__(self, n_estimators=100, random_state=None):
        self.n_estimators = n_estimators
        self.random_state = random_state
    
    def fit(self, X):
        random_state = check_random_state(self.random_state)
        seeds = random_state.randint(MAX_INT, size=self.n_estimators)

        self.forest = [RCTree(X, random_state=seed)
            for i, seed in enumerate(seeds)]
        self.leaves = self.forest[0].leaves
        
#     def decision_function(self, log=lambda x: x):
#         codisp = defaultdict(float)
#         for tree in self.forest:
#             for leaf in tree.leaves:
#                 codisp[leaf] += log(tree.codisp(leaf))
#         for leaf in codisp:
#             codisp[leaf] /= len(codisp)
#         return - np.asarray(sorted(codisp.items()))[:,1]

    def codisp_leaves(self, X):
        scores = np.array([self._compute_codisp_a_leaf(p)
            for p in X])
        return scores

    def log_codisp_leaves(self, X):
        scores = np.array([self._compute_codisp_a_leaf(p, log=np.log1p)
            for p in X])
        return scores

    def _compute_codisp_a_leaf(self, p, log=lambda x: x):
        score = sum([log(tree.codisp(tree.query(p)))
            for tree in self.forest]) / self.n_estimators
        return score

    def codisp_samples(self, X, L=None):
        scores = np.array([self._compute_codisp_a_sample(p)
            for p in X])
        return scores

    def log_codisp_samples(self, X, L=None):
        scores = np.array([self._compute_codisp_a_sample(p, log=np.log1p)
            for p in X])
        return scores

    def _compute_codisp_a_sample(self, p, log=lambda x: x):
        score = sum([log(self._compute_codisp_after_insert_a_sample(p, tree))
            for tree in self.forest]) / self.n_estimators
        return score
    
    def _compute_codisp_after_insert_a_sample(self, p, tree):
        leaf = tree.insert_point(p, index='new')
        codisp = tree.codisp(leaf)
        tree.forget_point('new')
        return codisp
