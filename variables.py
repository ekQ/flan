"""
Class Problem: parameters of the optimization problem
Class W: implementation of the linearized quadratic term w
"""
import numpy as np
from scipy import sparse


class Problem:
    """
    Parameters of the problem.
    """
    def __init__(self, f, D, g, gap_cost, N, A, candidate_matches, g2n, gt=None,
                 B=None, self_discount=True, max_entities=None):
        """
        Input:
            f -- Cost of opening an entity.
            D -- (Sparse) distance matrix.
            g -- Discount for matching neighbors to neighbors.
            gap_cost -- Cost for matching a node to itself (=to a gap)
            N -- Total number of items in all input graphs.
            A -- Overall adjacency matrix (sparse block matrix containing all
                 input graphs).
            candidate_matches -- Dict with items as keys and candidate match
                                 lists as values.
            g2n -- Graph-to-item dict.
            gt -- Groundtruth solution
            B -- Underlying entity graph (if None, then B = A).
            self_discount -- Whether to give the pairwise discount in case
                             neighbors are mapped to themselves.
            max_entities -- Fixed number of entities.
        """
        self.f = f
        self.D = D
        self.g = g
        self.gap_cost = gap_cost
        self.N = N
        self.A = A
        self.AA = None
        self.adj_list = self.construct_adjacency_lists()
        if B is not None:
            self.B = B
        else:
            self.B = A.copy()
        self.candidate_matches = candidate_matches
        self.self_discount = self_discount
        # {item A: set of other items that could be matched to A}
        # (might be = candidate_matches)
        self.candidates_to_match = self.construct_from_matches()
        self.groundtruth = gt
        self.init_w()
        #self.init_diff_w(self.A)
        self.graph2nodes = g2n
        self.node2graph = {}
        for graph, nodes in self.graph2nodes.iteritems():
            for node in nodes:
                self.node2graph[node] = graph
        self.max_entities = max_entities
        self.fix_max_entities = (max_entities is not None)

    def construct_adjacency_lists(self):
        adj_list = []
        print "N", self.N
        for i in range(self.N):
            if sparse.issparse(self.A):
                neighs = list(self.A[i].nonzero()[1])
            else:
                neighs = list(np.nonzero(self.A[i])[0])
            adj_list.append(neighs)
        return adj_list

    def construct_from_matches(self):
        candidates_to_match = {i: set() for i in range(self.N)}
        for i, candidates in enumerate(self.candidate_matches):
            for j in candidates:
                candidates_to_match[j].add(i)
        return candidates_to_match

    def init_w(self):
        """
        Find all "squares", i.e. neighbors that can be mapped to neighbors,
        corresponding to the w variables.
        """
        n_w = 0
        self.squares = [] # (i,j,k,l)
        for i, i_neighs in enumerate(self.adj_list):
            for j in self.candidate_matches[i]:
                if not self.self_discount and j == i:
                    continue
                for k in i_neighs:
                    for l in self.candidate_matches[k]:
                        if not self.self_discount and l == k:
                            continue
                        if self.B[j, l] > 0:
                            self.squares.append((i, j, k, l))
                            n_w += 1
        print "{} w's.".format(n_w)

    def init_diff_w(self, B):
        n_w = 0
        self.squares = [] # (i,j,k,l)
        for i, i_matches in enumerate(self.candidate_matches):
            for k, k_matches in enumerate(self.candidate_matches):
                for j in i_matches:
                    for l in k_matches:
                        diff = (self.A[i, k] - B[j, l])**2
                        if diff > 0:
                            self.squares.append((i, j, k, l))#,diff))
                            n_w += 1
        print "{} diff w's.".format(n_w)

    def construct_AA(self, x_start_idxs, cm_dicts, xlen):
        """
        Construct A "Kronecker product" A matrix for a given blocking
        and edgelist.
        """
        if self.candidate_matches is None:
            return sparse.kron(self.A, self.A)
        else:
            # Compute node -> neighbor set structure
            adj_set = []
            assert self.N == self.A.shape[0], "%d != %d" % (self.N,
                                                            self.A.shape[0])
            for i in range(self.N):
                if sparse.issparse(self.A):
                    adj_set.append(set(self.A[i].nonzero()[1]))
                else:
                    adj_set.append(set(np.nonzero(self.A[i])[0]))
            row_ind = []
            col_ind = []
            for i in range(self.N):
                for j in self.candidate_matches[i]:
                    for k in adj_set[i]:
                        oks = adj_set[j] & set(self.candidate_matches[k])
                        for ok_idx in oks:
                            row = x_start_idxs[i] + cm_dicts[i][j]
                            col = x_start_idxs[k] + cm_dicts[k][ok_idx]
                            row_ind.append(row)
                            col_ind.append(col)
            self.AA = sparse.csr_matrix((np.ones(len(col_ind)),
                                         (row_ind, col_ind)),
                                        shape=(xlen, xlen), dtype=np.int32)


class W:
    """
    Implementation of the linearized quadratic term w.
    """
    def __init__(self, P):
        """
        Input:
            adj_list -- adjacency list
        """
        self.w = {} # (i,j) -> k -> l -> 0/1
        for i, j, k, l in P.squares:
            if (i, j) not in self.w:
                self.w[(i, j)] = {}
            if k not in self.w[(i, j)]:
                self.w[(i, j)][k] = {}
            self.w[(i, j)][k][l] = 0
        self.P = P
    
    def get(self, i, j, k, l):
        if self.is_initialized(i, j, k, l):
            return self.w[(i, j)][k][l]
        else:
            return 0

    def is_initialized(self, i, j, k, l):
        if (i, j) not in self.w:
            return False
        if k not in self.w[(i, j)]:
            return False
        if l not in self.w[(i, j)][k]:
            return False
        return True

    def set(self, i, j, k, l):
        self.w[(i, j)][k][l] = 1

    def unset(self, i, j, k, l):
        self.w[(i, j)][k][l] = 0

    def iter(self, fixed_ij=None, fixed_k=None, graph=None):
        """
        Iterate over values of w.

        Input:
            fixed_ij -- 2-tuple fixing the values of i and j (optional)
            fixed_k -- fixing the value of k
            graph -- iterate only over i's from this graph (optional)

        Ouput:
            An iterator yielding i, j, k, l, and the value of w at those
            indices.
        """
        if fixed_ij is not None:
            assert len(fixed_ij) == 2, "Bad ij argument: {}".format(fixed_ij)
            if fixed_ij not in self.w:
                ijs = []
            else:
                ijs = [fixed_ij]
        else:
            ijs = self.w.iterkeys()
        for (i, j) in ijs:
            if graph is not None and self.P.node2graph[i] != graph:
                continue
            if fixed_k is not None:
                kitems = [(fixed_k, self.w[(i, j)].get(fixed_k, []))]
            else:
                kitems = self.w[(i, j)].iteritems()
            for k, ls in kitems:
                for l in ls.iterkeys():
                    val = self.w[(i, j)][k][l]
                    yield i, j, k, l, val

    def iter_nonzero(self, ij=None, k=None):
        """
        Iterate over nonzero values of w and yield the indices (i,j,k,l).
        """
        for i, j, k, l, val in self.iter(ij, k):
            if val == 1:
                yield i, j, k, l
