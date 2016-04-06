"""
Class Assignment for representing a multiple network alignment.
NOTE that this class is mainly used with baseline methods like 'agglomerative'.
"""
import numpy as np
from scipy import sparse
import random
from copy import deepcopy


class Assignment:
    def __init__(self, n, matches, candidate_matches=None,
                 candidate_matches_dicts=None):
        """
        Input:
            n -- number of items
            matches -- initial list of matching object indices
            candidate_matches -- list of lists: item -> list of candidate
                                 matches
            candidate_matches_dicts -- list of dict: item -> candidate match ->
                                       idx of the cm

        For example, if candidate_matches = [[0,2], [1,3]], then
        candidate_matches_dicts = [{0:0, 2:1}, {1:0, 3:1}]
        """
        self.n = n
        self.matches = np.asarray(matches, dtype=int)
        self.candidate_matches = candidate_matches
        self.candidate_matches_dicts = candidate_matches_dicts
        if self.candidate_matches is not None:
            # Number of (item, candidate match) pairs
            self.xlen = 0
            # Start indices of the matches of different items
            self.x_start_idxs = []
            for i in range(n):
                self.xlen += len(self.candidate_matches[i])
                if i == 0:
                    self.x_start_idxs.append(0)
                else:
                    self.x_start_idxs.append(self.x_start_idxs[-1] + 
                                             len(self.candidate_matches[i-1]))
            if self.candidate_matches_dicts is None:
                # Construct the dicts
                self.candidate_matches_dicts = []
                for i in range(n):
                    self.candidate_matches_dicts.append(
                            {cm:j for j,cm in enumerate(self.candidate_matches[i])})
                    # Check that matches are in candidates
                    assert matches[i] in self.candidate_matches_dicts[i], \
                            "matches[%d]=%d not a candidate (%s)" % \
                            (i, matches[i], str(self.candidate_matches_dicts[i]))
        self.clusters = self.get_clusters()

    def get_clusters(self):
        clusters = {}
        for i, clust in enumerate(self.matches):
            if clust in clusters:
                clusters[clust].append(i)
            else:
                clusters[clust] = [i]
        return clusters

    def get_cluster_keys(self, shuffle=False):
        cks = list(set(self.matches))
        #cks = self.clusters.keys()
        if shuffle:
            random.shuffle(cks)
        return cks

    def get_candidates(self, clust):
        """
        Get a list of candidate matches for cluster clust.
        """
        if self.candidate_matches_dicts is not None:
            return self.candidate_matches_dicts[clust].keys()
        else:
            return range(self.n)

    def copy(self):
        # NB: candidate_matches should be constant so no need to copy those
        return Assignment(self.n, deepcopy(self.matches),
                          self.candidate_matches,
                          deepcopy(self.candidate_matches_dicts))

    def count_opened_entities(self):
        return len(set(self.matches))

    def construct_Xx(self):
        n = self.n
        X = sparse.csr_matrix((np.ones(n),
            (range(n),self.matches)), shape=(n,n), dtype=np.bool_)
        if self.candidate_matches is not None:
            # Collect x indices
            x_idxs = []
            for i in range(n):
                #print self.candidate_matches_dicts[i], self.matches[i], i
                new_idx = self.x_start_idxs[i] + self.candidate_matches_dicts[i][self.matches[i]]
                if new_idx >= self.xlen:
                    print "Too big:", self.xlen, self.x_start_idxs[i], \
                        self.candidate_matches_dicts[i][self.matches[i]], i, \
                        self.candidate_matches_dicts[i]
                x_idxs.append(new_idx)
            x = sparse.csr_matrix((np.ones(n), (x_idxs, np.zeros(n))),
                                  shape=(self.xlen,1), dtype=np.bool_)
        else:
            x = sparse.csr_matrix((np.ones(n), 
                (np.arange(0,n*n,n) + np.asarray(self.matches), np.zeros(n))),
                shape=(n*n,1), dtype=np.bool_)
        return X, x

    def merge_clusters(self, src_clust, dst_clust):
        # Update candidate matches
        if self.candidate_matches is not None:
            assert dst_clust in self.candidate_matches_dicts[src_clust], \
                    "Not a candidate cluster %s, %s" % (str(dst_clust),
                    str(self.candidate_matches_dicts[src_clust]))
            new_cmd = {}
            for dst_cand in self.candidate_matches_dicts[dst_clust].iterkeys():
                if dst_cand in self.candidate_matches_dicts[src_clust]:
                    # Keep only the candidates which are candidates also for
                    # the cluster to be merged
                    new_cmd[dst_cand] = self.candidate_matches_dicts[dst_clust][dst_cand]
            self.candidate_matches_dicts[dst_clust] = new_cmd
        # Update matches
        self.update_all_matches(src_clust, dst_clust)
        #for i, clust in enumerate(self.matches):
        #    if clust == src_clust:
        #        self.matches[i] = dst_clust

    def update_match(self, idx, dst_clust):
        """
        Update the assigned entity for given idx. idx can be a list of indices.
        """
        self.clusters[self.matches[idx]].remove(idx)
        if dst_clust in self.clusters:
            self.clusters[dst_clust].append(idx)
        else:
            self.clusters[dst_clust] = [idx]
        self.matches[idx] = dst_clust

    def update_all_matches(self, src_clust, dst_clust, do_assert=False):
        """
        Update entity assignments for all items assigned to src_clust.

        Return the updated indices.
        """
        src_idxs = np.nonzero(self.matches == src_clust)[0]
        if do_assert:
            for idx in src_idxs:
                assert dst_clust in self.candidate_matches_dicts[idx], \
                    str((idx, dst_clust, self.candidate_matches_dicts[idx]))
        self.update_match(src_idxs, dst_clust)
        return src_idxs
