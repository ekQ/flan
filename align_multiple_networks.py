"""
Interface for running the different multiple network alignment methods.
"""
import numpy as np
import random
import time
from scipy import sparse
import networkx as nx
import argparse

import variables
from assignment import Assignment
from baselines import agglomerative as aggl
from baselines import icm
from baselines import isorankn
from baselines import random_alignment as randa
from lagrangian_relaxation import flan as LD
from lagrangian_relaxation import natalie as mklau


def align_multiple_networks(
        Gs, cost_params, similarity_tuples=None, method="LD", max_iters=100,
        max_algorithm_iterations=1, max_entities=None, shuffle=False,
        self_discount=True, do_evaluate=True, u_w=None):
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
        cost_params -- dict specifying cost 'f', discount 'g', and 'gap_cost'
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
    gao = range(len(Gs))
    if method == 'isorankn':
        gao = None
    A, B, similarities, GG, groundtruth, graph2nodes, old2new_label = \
        construct_A_and_sims(Gs, similarity_tuples, graph_alignment_order=gao,
                             shuffle_node_labels=shuffle)
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
        matches.append(cms[random.randint(0, len(cms)-1)])
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

    # Groundtruth cost
    if groundtruth is not None:
        gt_c_pay = compute_score(P, groundtruth, True)
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
        # Import prog only here to avoid problems with circular imports
        from baselines import progressive_multiple_alignment as prog
        results, master_G = prog.progressive_multiple_alignment(
            Gs, cost_params, similarity_tuples, sub_method, max_iters,
            max_algorithm_iterations, max_entities, shuffle, self_discount,
            do_evaluate, update_edges)
        GG = master_G
    elif method in ("LD", "flan"):
        results = LD.LD_algorithm(P, max_iters, max_algorithm_iterations)
    elif method == "binB-LD":
        results = LD.LD_algorithm(P, max_iters, max_algorithm_iterations,
                                  binary_B=True)
    elif method == "clusterLD":
        results = LD.LD_algorithm(P, max_iters, max_algorithm_iterations,
                                  cluster=True)
    elif method == "LDunary":
        results = LD.LD_algorithm(P, use_binary=False)
    elif method in ("Klau", "mKlau", "natalie"):
        results = mklau.klaus_algorithm(P, max_iters, u_w=u_w)
    elif method == 'isorankn':
        results = isorankn.align_isorankn(P)
    elif method == 'rand':
        results = randa.align_randomly(P)
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

    if shuffle:
        # Deshuffle the results
        x, GG, graph2nodes, prev_xs = deshuffle(
            old2new_label, x, GG, graph2nodes, prev_xs)

    if do_evaluate:
        other['prev_scores'] = []
        print
        for ai, x in enumerate(prev_xs):
            precision, recall, fscore = evaluate_alignment_fast(x, GG)
            other['prev_scores'].append((precision, recall, fscore))
            print "Algorithm iteration {}: p={:.3f}, r={:.3f}, f1={:.3f}\n".format(
                    ai, precision, recall, fscore)
        other['scores'] = other['prev_scores'][-1]
        #write_alignment_results(x, GG, 'output_clusters2.txt')
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
    A = nx.to_scipy_sparse_matrix(fullG, weight=None, format='dok')
    B = nx.to_scipy_sparse_matrix(fullG, weight='weight', format='dok')
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
        score += P.D[i, j]
        unary_score += P.D[i, j]
        for k in P.adj_list[i]:
            l = x[k]
            Bjl = P.B.get(j, l)
            if Bjl:
                score -= Bjl * P.g / 2.0
                binary_score -= Bjl * P.g / 2.0
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


def write_alignment_results(x, master_G, output):
    clusters = {}
    for xi, val in enumerate(x):
        if val not in clusters:
            clusters[val] = []
        clusters[val].append((master_G.node[xi]['subgraph'],
                              master_G.node[xi]['id']))

    lines = []
    for clust in clusters.itervalues():
        clust = sorted(["{}_{}".format(*c) for c in clust])
        lines.append("{}\n".format(" ".join(clust)))
    lines = sorted(lines)
    with open(output, 'w') as fout:
        for line in lines:
            fout.write(line)
    print "\nWrote the alignment to:", output


def read_problem_files(edgefile, similarityfile):
    """
    Read alignment problem instance from files and construct suitable data
    structures.

    Input:
        edgefile --  Path to file containing edges of all the input graphs
        similarityfile -- Path to file containing candidate matches and
                          similarities of vertices

    Output:
        Gs -- list of (networkx) input graphs
        similarity_tuples
    """
    Gd = {}
    fe = open(edgefile, 'r')
    for line in fe:
        parts = line.strip().split()
        if len(parts) < 3 or len(parts) > 4:
            print "Malformed line:\n{}".format(line)
            continue
        gid = parts[0]
        if gid not in Gd:
            Gd[gid] = nx.Graph()
        u = parts[1]
        v = parts[2]
        if not Gd[gid].has_node(u):
            Gd[gid].add_node(u, id=u, subgraph=gid)
        if not Gd[gid].has_node(v):
            Gd[gid].add_node(v, id=v, subgraph=gid)
        if len(parts) == 3:
            weight = 1
        else:
            weight = float(parts[3])
        Gd[gid].add_edge(u, v, weight=weight)
    fe.close()

    tups = []
    fs = open(similarityfile, 'r')
    for line in fs:
        parts = line.strip().split()
        if len(parts) != 5:
            print "Malformed sf line:\n{}".format(line)
            continue
        gid1 = parts[0]
        u1 = parts[1]
        gid2 = parts[2]
        u2 = parts[3]
        sim = float(parts[4])
        if gid1 not in Gd:
            Gd[gid1] = nx.Graph()
        if gid2 not in Gd:
            Gd[gid2] = nx.Graph()
        Gd[gid1].add_node(u1, id=u1, subgraph=gid1)
        Gd[gid2].add_node(u2, id=u2, subgraph=gid2)
        tups.append((gid1, u1, gid2, u2, sim))
    fs.close()

    # Order input graphs and assign index number to each
    Gi = sorted(Gd.items())
    print "\nThe number of input graphs is: {}".format(len(Gi))
    print "The number of nodes per graph:"
    for gid, G in Gi:
        print "\tGraph {}: {}".format(gid, len(G))
    print
    gids, Gs = zip(*Gi)
    Gmap = {gid: i for i, gid in enumerate(gids)}
    new_tups = []
    for gid1, u1, gid2, u2, sim in tups:
        new_tups.append((Gmap[gid1], u1, Gmap[gid2], u2, sim))
    return Gs, new_tups


def main():
    random.seed(14564356)
    np.random.seed(145654)
    parser = argparse.ArgumentParser()
    parser.add_argument("edgefile", help=
        """
        Path to file containing the edges of the input graphs. One row
        contains one edge with the format: "graphID edgeID1 edgeID2 (weight)",
        where edge weight is optional (1 by default). NOTE: currently edge
        weights are NOT supported.
        """
    )
    parser.add_argument("similarityfile", help=
        """
        Path to file containing the candidate matches for vertices and the
        associated similarity values. One row contains one candidate match of
        a node:
            "graphID1 nodeID1 graphID2 nodeID2 similarity".
        Note that these matches are directed (add nodeID2 -> nodeID1
        separately) if you want. Also note that a node can be matched with
        only the nodes specified in this file so typically you want to add
        at least the option to map to itself ("graphID1 nodeID1 graphID1
        nodeID1 1").
        """
    )
    method_map = {"flan": "LD", "flan0": "LD", "cflan": "LD",
                  "prognatalie": "progmKlau", "prognatalie++": "upProgmKlau",
                  "natalie": "mKlau", "icm": "ICM", "isorankn": "isorankn"}
    parser.add_argument("-m", "--method", default="flan",
                        choices=sorted(method_map.keys()), help=
                        "Alignment method. Default: flan")
    parser.add_argument("-f", type=float, default=1, help=(
        "Cost of opening an entity. A larger value results in fewer clusters. "
        "Default: 1"))
    parser.add_argument("-g", type=float, default=0.5, help=(
        "Discount for mapping neighbors to neighbors. Default: 0.5"))
    parser.add_argument("-e", "--entities", type=int, help=(
        "Number of entities (must be specified when cFLAN is used"))
    parser.add_argument("-i", "--iterations", type=int, default=300, help=
        "Number of iteration the Lagrange multipliers are solved. Default: 300")
    parser.add_argument("-o", "--output", default="output_clusters.txt", help=(
        "Output filename. Each line of the output contains a list of "
        "graphID_nodeID pairs aligned to the same cluster. Default: "
        "output_clusters.txt"))
    args = parser.parse_args()
    Gs, similarity_tuples = read_problem_files(args.edgefile,
                                               args.similarityfile)
    method = method_map[args.method]
    mai = 1
    if args.method in ("flan", "cflan"):
        mai = 5
    cost_params = {"f": args.f, "g": args.g, "gap_cost": args.f}
    max_entities = args.entities
    max_iters = args.iterations
    x, other = align_multiple_networks(Gs, cost_params, similarity_tuples,
                                       method, max_iters, mai, max_entities,
                                       shuffle=True, self_discount=True,
                                       do_evaluate=False)
    write_alignment_results(x, other['G'], args.output)

if __name__ == "__main__":
    main()
