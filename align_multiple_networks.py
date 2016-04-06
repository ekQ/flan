"""
Interface for running the different multiple network alignment methods.
"""
import numpy as np
import random
import time
from scipy import sparse
import networkx as nx

import variables
from assignment import Assignment
from baselines import agglomerative as aggl
from baselines import icm
from baselines import progressive_multiple_alignment as prog
from lagrangian_relaxation import flan as LD
from lagrangian_relaxation import natalie as mklau


def align_multiple_networks(Gs, cost_params, similarity_tuples=None,
                            method="LD", max_iters=100,
                            max_algorithm_iterations=1, max_entities=None,
                            shuffle=False, self_discount=True):
    """
    Compute the best alignment for the given networks. You should either provide
    argument 'similarity_tuples' or otherwise the graph nodes should have the
    following attributes:
        name -- Nodes with the same 'name' from different input graphs are
                considered as candidate entities
        subgraph -- Indicates the input graph index of the node
        entity -- (OPTIONAL) Groundtruth cluster index of the node

    Input:
        Gs -- list of NetworkX input graphs
        cost_params -- dict specifying adjacency matrix 'A', cost 'f', discount
                       'g', and cost 'gap_cost'
        similarity_tuples -- (OPTIONAL) list of tuples for candidate matches:
                                        (graph1_index, node1, graph2_index,
                                         node2, similarity)
        method -- the method used for merging
        shuffle -- whether to shuffle node labels (might affect if there are
                   ties)

    Output:
        matched indices for each object
        score
    """
    A, B, similarities, GG, groundtruth, graph2nodes, old2new_label = \
        construct_A_and_sims(Gs, similarity_tuples, graph_alignment_order=
                             range(len(Gs)), shuffle_node_labels=shuffle)
    n = A.shape[0]
    # Initialize objects to themselves
    matches = []
    candidate_matches = []
    n_xij = 0
    for i, block_dict in enumerate(similarities):
        cms = block_dict.keys()
        assert len(cms) > 0, "Each node should have at least 1 candidate " + \
                             "match, node {} has zero.".format(i)
        candidate_matches.append(cms)
        matches.append(cms[random.randint(0,len(cms)-1)])
        n_xij += len(cms)
    print "Total number of xij's:", n_xij
    asg = Assignment(n, matches, candidate_matches)
    print "Constructing Kronecker product of size %d x %d..." % (asg.xlen,
                                                                 asg.xlen)
    t0 = time.time()
    print "Constructed in %.2f seconds." % (time.time() - t0)
    # TODO Change the D matrix into a dict and wrap it inside a class
    D = sparse.lil_matrix((n, n))
    for i in range(len(similarities)):
        for j, sim in similarities[i].iteritems():
            D[i, j] = 1 - sim

    f = cost_params['f']
    g = cost_params['g']
    gap_cost = cost_params.get('gap_cost', None)
    P = variables.Problem(f, D, g, gap_cost, n, A, candidate_matches,
                          graph2nodes, B=B, self_discount=self_discount,
                          max_entities=max_entities)
    # NOTE This function call is only needed for computing groundtruth score
    # but it could be optimized away
    P.construct_AA(asg.x_start_idxs, asg.candidate_matches_dicts, asg.xlen)

    # Groundtruth cost
    if groundtruth is not None:
        gt_c = compute_score(P, groundtruth, False)
        gt_c_pay = compute_score(P, groundtruth, True)
        print "Groundtruth score (f=0):", gt_c
        print "Groundtruth score (f={}): {}".format(P.f, gt_c_pay)

    other = {"iterations": -1, "cost": 0, "lb": -1, "ub": -1}
    if method == "agglomerative":
        asg = aggl.agglomerative(asg, similarities, P, max_iters)
    elif method == "agglomerative_fixed":
        asg = aggl.agglomerative_fixed(asg, similarities, P, max_entities)
    elif method == "ICM":
        asg, cost, iteration = icm.ICM(asg, similarities, P, max_iters)
        other["cost"] = cost
        other["iterations"] = iteration
    elif method.startswith('prog') or method.startswith('upProg'):
        if method.startswith('up'):
            update_edges = True
            sub_method = method[6:]
        else:
            update_edges = False
            sub_method = method[4:]
        results, master_G = prog.progressive_multiple_alignment(
            Gs, cost_params, similarity_tuples, sub_method, max_iters,
            max_algorithm_iterations, max_entities, shuffle, self_discount,
            update_edges)
        GG = master_G
    elif method in ("LD", "FLAN"):
        results = LD.LD_algorithm(P, max_iters, max_algorithm_iterations)
    elif method == "binB-LD":
        results = LD.LD_algorithm(P, max_iters, max_algorithm_iterations,
                                  binary_B=True)
    elif method == "clusterLD":
        results = LD.LD_algorithm(P, max_iters, max_algorithm_iterations,
                                  cluster=True)
    elif method == "LDunary":
        results = LD.LD_algorithm(P, use_binary=False)
    elif method in ("Klau", "mKlau", "Natalie"):
        results = mklau.klaus_algorithm(P, max_iters)
    else:
        print "Unknown method:", method
        raise Exception("Unknown method")

    if method not in ["agglomerative", "agglomerative_fixed", "ICM"]:
        asg.matches = np.asarray(results['best_x'], dtype=int)
        other = results
        other["cost"] = results["feasible_scores"][-1]

    x = asg.matches
    if 'prev_xs' in other:
        prev_xs = other['prev_xs']
    else:
        prev_xs = [x]

    x_c = compute_score(P, x, False)
    x_c_pay = compute_score(P, x, True)
    print "best_x score (f=0):", x_c
    print "best_x score (f={}): {}".format(P.f, x_c_pay)

    if shuffle:
        # Deshuffle the results
        x, GG, graph2nodes, prev_xs = deshuffle(
            old2new_label, x, GG, graph2nodes, prev_xs)

    other['prev_scores'] = []
    print
    for ai, x in enumerate(prev_xs):
        precision, recall, fscore = evaluate_alignment_fast(x, GG)
        other['prev_scores'].append((precision, recall, fscore))
        print "Algorithm iteration {}: p={:.3f}, r={:.3f}, f1={:.3f}\n".format(
                ai, precision, recall, fscore)
    other['scores'] = other['prev_scores'][-1]
    other['G'] = GG
    other['graph2nodes'] = graph2nodes
    other['n_clusters'] = len(asg.get_clusters())
    return x, other


def construct_A_and_sims(Gs, similarity_tuples=None, graph_alignment_order=None,
                         shuffle_node_labels=True):
    """
    Input:
        Gs -- list of networkx graphs. The nodes of the graphs should have the
              following attributes:
                * subgraph -- input graph index
                * entity -- index of the underlying entity (TODO: consider
                            removing this attribute)
                * name -- (ONLY if similarity_tuples argument not provided) a
                          label for the node so that it will get a large
                          similarity with the other nodes having the same label
                          and otherwise zero
        similarity_tuples -- (OPTIONAL) list of tuples:
                                        (graph1_index, node1, graph2_index,
                                         node2, similarity)
        graph_alignment_order -- In which order are graphs aligned (defines an
                                 ordering for graphs to avoid trying to match
                                 both i->j and j->i).
    """
    assert graph_alignment_order is None or len(Gs)==len(graph_alignment_order)
    node2graph = {}
    graph2nodes = {}
    fullG = nx.Graph()
    node2idx = {}
    for graph_idx, inputG in enumerate(Gs):
        len_prev = len(fullG)
        fullG = nx.disjoint_union(fullG, inputG)
        new_labels = range(len_prev, len(fullG))
        for lab in new_labels:
            node2graph[lab] = graph_idx
        graph2nodes[graph_idx] = new_labels
        for node, idx in zip(inputG.nodes(), new_labels):
            # Map graph-node tuple to the new index
            node2idx[(graph_idx, node)] = idx
    old2new_label = None
    if shuffle_node_labels:
        # Shuffle node labels just in case
        renamed_labels = np.random.permutation(len(fullG))
        old2new_label = {i: renamed_labels[i] for i in fullG.nodes()}
        fullG = nx.relabel_nodes(
            fullG, old2new_label, copy=True)
        node2graph = {renamed_labels[i]: graph for i, graph in
                      node2graph.iteritems()}
        for graph, nodes in graph2nodes.iteritems():
            graph2nodes[graph] = [renamed_labels[i] for i in nodes]
        for key, old_idx in node2idx.iteritems():
            node2idx[key] = renamed_labels[old_idx]
    A = nx.to_scipy_sparse_matrix(fullG, weight=None)
    B = nx.to_scipy_sparse_matrix(fullG, weight='weight')
    #print "\n\nDiff elements:", (B-A).sum()
    #print "Max B element:", B.max()

    # Similarities
    if similarity_tuples is None:
        sims = []
        gts = []
        for i in fullG.nodes():
            s = {}
            i_gt = i
            for j in fullG.nodes():
                # A sanity check
                if fullG.node[i]['subgraph'] != fullG.node[j]['subgraph'] and \
                        (node2graph[i] == node2graph[j]):
                    print("Something wrong with subgraphs:",
                          fullG.node[i]['subgraph'], fullG.node[j]['subgraph'],
                          node2graph[i], node2graph[j])
                    assert False
                gi = fullG.node[i]['subgraph']
                gj = fullG.node[j]['subgraph']
                order_ok = True
                if graph_alignment_order is not None:
                    gi_idx = graph_alignment_order.index(gi)
                    gj_idx = graph_alignment_order.index(gj)
                    order_ok = gi_idx >= gj_idx
                if fullG.node[i]['name'] == fullG.node[j]['name'] and \
                        (i == j or gi != gj) and order_ok:
                    s[j] = 1#0.99 + random.random()*0.01
                if 'entity' in fullG.node[i] and 'entity' in fullG.node[j] and \
                        fullG.node[i]['entity'] == fullG.node[j]['entity'] and \
                        i_gt is None:
                    i_gt = j
            sims.append(s)
            if i_gt is None:
                print "Warning: Adding a None value to the groundtruth vector."
            gts.append(i_gt)
    else:
        sims = []
        for i in range(len(fullG)):
            sims.append({})
        for sim_tup in similarity_tuples:
            v1 = node2idx[(sim_tup[0], sim_tup[1])]
            v2 = node2idx[(sim_tup[2], sim_tup[3])]
            sim = sim_tup[4]
            g1 = node2graph[v1]
            g2 = node2graph[v2]
            order_ok = True
            if graph_alignment_order is not None:
                gi_idx = graph_alignment_order.index(g1)
                gj_idx = graph_alignment_order.index(g2)
                order_ok = gi_idx >= gj_idx
            if order_ok:
                sims[v1][v2] = sim
        for i in range(len(fullG)):
            if len(sims[i]) == 0:
                # Each node should have at least one candidate match
                sims[i][i] = 1
        # Set groundtruths
        t0 = time.time()
        gts = []
        for i in fullG.nodes():
            gts.append(i)
        print "Groundtruth constructed ({:.2f} seconds).".format(time.time()-t0)
    return A, B, sims, fullG, gts, graph2nodes, old2new_label


def deshuffle(old2new_label, x, G, graph2nodes, prev_xs=None):
    new2old_label = {val: key for key, val in old2new_label.iteritems()}
    new_x = [None] * len(x)
    for xi, val in enumerate(x):
        new_x[new2old_label[xi]] = new2old_label[val]
        #new_x[xi] = new2old_label[val]
    new_prev_xs = None
    if prev_xs is not None:
        new_prev_xs = []
        for px in prev_xs:
            new_px = [None] * len(px)
            for xi, val in enumerate(px):
                new_px[new2old_label[xi]] = new2old_label[val]
            new_prev_xs.append(new_px)
    new_G = nx.relabel_nodes(G, new2old_label, copy=True)
    for graph, nodes in graph2nodes.iteritems():
        graph2nodes[graph] = [new2old_label[i] for i in nodes]
    new_graph2nodes = graph2nodes
    return new_x, new_G, new_graph2nodes, new_prev_xs


def evaluate_alignment_fast(x, GG):
    # Evaluate: count number of true pairs
    cm = np.zeros((2, 2), dtype=int)  # Confusion matrix
    n = len(x)
    cm[0, 0] = (n**2 - n) / 2

    match_set = set()
    ideal_set = set()
    match_clusters = {}
    ideal_clusters = {}

    for i, mi in enumerate(x):
        if mi not in match_clusters:
            match_clusters[mi] = []
        match_clusters[mi].append(i)
    for members in match_clusters.itervalues():
        for i in members:
            for j in members:
                if j <= i:
                    continue
                match_set.add((i, j))

    for i in range(len(GG)):
        mi = GG.node[i]['entity']
        if mi not in ideal_clusters:
            ideal_clusters[mi] = []
        ideal_clusters[mi].append(i)
    for members in ideal_clusters.itervalues():
        for i in members:
            for j in members:
                if j <= i:
                    continue
                ideal_set.add((i, j))
    #print "Match set:", match_set
    #print "Ideal set:", ideal_set
    tp = len(match_set & ideal_set)
    fp = len(match_set - ideal_set)
    fn = len(ideal_set - match_set)
    cm[0, 1] = fp
    cm[1, 0] = fn
    cm[1, 1] = tp
    cm[0, 0] -= fp + fn + tp
    print "True number of cluster {}, obtained number of clusters {}.".format(
        len(ideal_clusters), len(match_clusters)
    )
    print cm

    precision = float(cm[1, 1]) / max(1, sum(cm[:, 1]))
    recall = float(cm[1, 1]) / max(1, sum(cm[1, :]))
    fscore = 2 * precision * recall / max(0.000000001, precision + recall)
    return precision, recall, fscore


def compute_score(P, x, pay4entities):
    n_opened = len(set(x))
    if not pay4entities:
        score = 0
    else:
        score = P.f * n_opened
    unary_score = 0
    binary_score = 0
    for i, j in enumerate(x):
        score += P.D[i,j]
        unary_score += P.D[i,j]
        for k in P.adj_list[i]:
            l = x[k]
            if P.B[j,l]:
                score -= P.B[j,l] * P.g / 2.0
                binary_score -= P.B[j,l] * P.g / 2.0
    print "    F:   {} opened entities, {} unary, {} binary, score {}.".format(
        n_opened, unary_score, binary_score, score)
    return score


def test():
    similarities = {0: {0: 1, 2: .9},
                    1: {1: 1, 3: .9},
                    2: {2: 1, 0: .9},
                    3: {3: 1, 1: .9},
                    }
    A = np.eye(4)
    A[0, 2] = 1
    A[2, 0] = 1
    A[1, 3] = 1
    A[3, 1] = 1
    print A
    f = 0.5
    g = 0.3
    x, oth = align_multiple_networks(similarities, A, f, g)
    print x
    print oth['min_cost']

if __name__ == "__main__":
    test()