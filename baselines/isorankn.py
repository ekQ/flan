"""
Wrapper class for IsoRankN binaries.

NB: Download the binaries from
    http://cb.csail.mit.edu/cb/mna/packages/isorank-n-v3-64.tar.gz
and place them in the baselines directory (see the subprocess call below).
"""
import os
import subprocess
import random


def align_isorankn(P):
    dir, problem_file = write_isorank_files(P)
    result_file = os.path.join(dir, "out3.dat")
    if P.f <= 1:
        alpha = P.f
    else:
        alpha = 1
    FNULL = open(os.devnull, 'w')
    subprocess.call(["./baselines/isorank-n-v3-64/isorank-n-v3-64", "--K", "30",
                     "--thresh", "1e-4", "--alpha", str(alpha), "--maxveclen",
                     "1000000", "-o", result_file, problem_file], stdout=FNULL,
                     stderr=subprocess.STDOUT)

    best_x = range(P.N)
    # Count the number of items assigned to candidate matches and non-candidates
    n_ok = 0
    n_bad = 0
    with open(result_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            ids = [int(val[1:]) for val in parts]
            cluster_id = min(ids)
            for id in ids:
                if cluster_id in P.candidate_matches[id] or \
                                id in P.candidate_matches[cluster_id]:
                    best_x[id] = cluster_id
                    n_ok += 1
                else:
                    # Find a candidate id within cluster
                    for id2 in ids:
                        if id2 in P.candidate_matches[id] or \
                                id in P.candidate_matches[id2]:
                            best_x[id] = id2
                            break
                    n_bad += 1
                #assert cluster_id in P.candidate_matches[id] or id in P.candidate_matches[cluster_id], "Bad isorankn assignment: {} to {}. {} and {}".format(id, cluster_id, P.candidate_matches[id], P.candidate_matches[cluster_id])
    print "\n\n{} out of {} OK.".format(n_ok, n_ok+n_bad)
    results = {"best_x": best_x, "iterations": -1, "cost": 0, "lb": -1,
               "ub": -1, "feasible_scores": [0]}
    return results


def write_isorank_files(P):
    print "Starting to write IsoRankN files..."
    dir = os.path.join('temp', '{}'.format(random.randint(0, 100000000)))
    if not os.path.exists(dir):
        os.makedirs(dir)
    meta_file = os.path.join(dir, 'data.inp')
    f_meta = open(meta_file, 'w')
    ng = len(P.graph2nodes)
    f_meta.write(dir + '\n-\n{}\n'.format(ng))

    # Edge files
    for i in range(ng):
        f_meta.write('G{}\n'.format(i))
        edge_file = os.path.join(dir, 'G{}.tab'.format(i))
        f_edge = open(edge_file, 'w')
        f_edge.write("INTERACTOR_A\tINTERACTOR_B\n")
        nodes = set(P.graph2nodes[i])
        for u, v, _ in P.A.iter():
            if u not in nodes or v not in nodes:
                continue
            #f_edge.write("v{}\tv{}\n".format(u, list(nodes)[random.randint(0, len(nodes)-1)]))
            f_edge.write("v{}\tv{}\n".format(u, v))
        f_edge.close()
    f_meta.close()

    # Unary files
    score0 = 1
    self_score = score0     # Similarity with item itself is 1
    scores = {}     # (Gi, Gj) -> list of matches and scores
    for i in range(ng):
        scores[(i, i)] = {}
        nodes1 = P.graph2nodes[i]
        for v in nodes1:
            scores[(i, i)][(v, v)] = self_score
    for u, vs in enumerate(P.candidate_matches):
        ug = P.node2graph[u]
        for v in vs:
            vg = P.node2graph[v]
            if ug < vg:
                u1 = u
                ug1 = ug
                u2 = v
                ug2 = vg
            else:
                u1 = v
                ug1 = vg
                u2 = u
                ug2 = ug
            if (ug1, ug2) not in scores:
                scores[(ug1, ug2)] = {}
            # Similarity between the items is 1-dist(u1,u2)
            scores[(ug1, ug2)][(u1, u2)] = score0 - P.D[u1, u2]
    for (g1, g2), candidates in scores.iteritems():
        unary_file = os.path.join(dir, 'G{}-G{}.evals'.format(g1, g2))
        f_unary = open(unary_file, 'w')
        for (u, v), score in candidates.iteritems():
            f_unary.write("v{}\tv{}\t{}\n".format(u, v, score))
        f_unary.close()
    print "Wrote."
    return dir, meta_file
