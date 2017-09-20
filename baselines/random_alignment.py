import random
import networkx as nx


def align_randomly(P):
    best_x = range(P.N)
    for u, vs in enumerate(P.candidate_matches):
        best_x[u] = vs[random.randint(0, len(vs)-1)]
    best_x = cluster_x(best_x, P.candidate_matches)
    results = {"best_x": best_x, "iterations": -1, "cost": 0, "lb": -1,
               "ub": -1, "feasible_scores": [0]}
    return results


def cluster_x(x, candidate_matches):
    """
    Assign all xs in a connected component to just a single node of
    that component.
    """
    G = nx.Graph()
    for i, j in enumerate(x):
        G.add_edge(i, j)
    components = nx.connected_components(G)
    new_x = [None] * len(x)
    for comp in components:
        # Component gets the label to which most of its members can be assigned
        best_label = -1
        best_count = -1
        for v in comp:
            count = 0
            for v2 in comp:
                if v in candidate_matches[v2]:
                    count += 1
            if count > best_count:
                best_count = count
                best_label = v
        for v in comp:
            new_x[v] = best_label
    return new_x
