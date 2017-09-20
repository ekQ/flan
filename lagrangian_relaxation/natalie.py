"""
Natalie modified to support multiple network alignment.
"""
import numpy as np
import time
from min_cost_matching import min_cost_matching
import variables
import networkx as nx

print_interval = 50
#print_interval = 1


def klaus_algorithm(P, max_iterations=500, use_binary=True, cluster=True,
                    u_w=None):
    for i in range(P.N):
        P.D[i, i] = P.gap_cost
    results = align_graphs(P, max_iterations, use_binary, cluster, u_w)
    return results


def align_graphs(P, max_iterations, use_binary, cluster, u_w=None):
    if u_w is None:
        u_w = {}
    l = -np.inf
    u = np.inf
    best_x = None
    best_x_iter = -1
    best_x_cands = None
    best_u_w = None
    theta = 0.1
    sg_N = 20
    sg_M = 10
    n = sg_N
    m = sg_M
    Zd_scores = []
    feasible_scores = []
    prev_duality_gap = np.inf
    n_equal_gaps = 0
    for t in range(0, max_iterations):
        has_converged, x, w, Zd_score, feasible_score, u_w, x_cands = \
                LD_iteration(P, u_w, theta, use_binary=use_binary)
        Zd_scores.append(Zd_score)
        feasible_scores.append(feasible_score)

        duality_gap = feasible_score - Zd_score
        rel_dg_change = np.abs(prev_duality_gap - duality_gap) \
                        / max(np.abs(feasible_score), np.abs(Zd_score))
        #rel_dg_change = np.abs(duality_gap) / max(np.abs(feasible_score),
        #                                          np.abs(Zd_score))
        if t % print_interval == 0:
            print "Duality gap {}, rel change {}".format(duality_gap,
                                                         rel_dg_change)
        #if np.abs(prev_duality_gap - duality_gap) < 1e-5:
        if rel_dg_change < 1e-3:
            n_equal_gaps += 1
            if n_equal_gaps == 5:
                print "Converged due to duality gap staying almost the same " \
                    + "for 5 iterations."
                has_converged = True
        else:
            n_equal_gaps = 0
        prev_duality_gap = duality_gap

        improved = False
        if Zd_score > l:
            l = Zd_score
            improved = True
            best_x_cands = x_cands
            best_u_w = u_w.copy()
        if feasible_score < u:
            u = feasible_score
            best_x = x
            best_x_iter = t
            improved = True
        if improved:
            m -= 1
            if m == 0:
                theta *= 2.0
                #print "Improved 10 times, setting theta to {}.".format(theta)
                m = sg_M
        else:
            n -= 1
            if n == 0:
                theta /= 2.0
                #print "Didn't improve for 20 times, setting theta to {}.".\
                #       format(theta)
                n = sg_N
        if t % print_interval == 0:
            print "{}: Zd score: {}, feasible score: {}".format(t, Zd_score,
                                                                feasible_score)
        #assert Zd_score - feasible_score < 1e-4, Zd_score - feasible_score
        if has_converged:
            print "*** Iteration converged in {} steps! ***".format(t+1)
            break
    print "Best feasible score {}, lb: {}, ub: {}, ub-iter: {}".format(
        u, l, u, best_x_iter)
    #print "Best x: {}".format(best_x)
    if cluster:
        best_x = cluster_x(best_x, P.candidate_matches)
    results = {'best_x': best_x, 'Zd_scores': Zd_scores,
               'feasible_scores': feasible_scores, 'iterations': t+1, 'lb': l,
               'ub': u, 'best_x_cands': best_x_cands, 'best_u_w': best_u_w}
    return results


def cluster_x(x, candidate_matches):
    """
    Assign all xs in a connected component to just a single node of
    that component.
    """
    G = nx.Graph()
    for i, j in enumerate(x):
        G.add_edge(i,j)
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


def LD_iteration(P, u_w, theta, use_binary=True):
    """
    Input:
        u_w -- lambdas for the w_ijkl = w_klij constraint

    Output:
        Optimal solution (x,y,w) and Lagrange multipliers
    """
    x, w, x_cands = solve_xw(P, u_w, use_binary=use_binary)
    feasible_score = compute_score(P, x, use_binary=use_binary)
    Zd_score = compute_infeasible_score(P, x, w, u_w, use_binary=use_binary)
    # Duality gap
    score_diff = feasible_score - Zd_score
    if score_diff < -1e-5:
        print "\nWarning!!! Negative duality gap: {}".format(score_diff)

    # Compute subgradient vectors
    sg = []
    for i, j, k, l, val in w.iter():
        sg.append(val - w.get(k, l, i, j))
    sg_norm = np.linalg.norm(sg, 2) ** 2
    if sg_norm < 1e-5:  # Converged
        return True, x, w, Zd_score, feasible_score, u_w, x_cands

    # Update Lagrange multipliers
    for i, j, k, l, val in w.iter():
        if (i, j, k, l) not in u_w:
            u_w[(i, j, k, l)] = 0
        sg = val - w.get(k, l, i, j)
        if sg_norm > 0:
            u_w[(i, j, k, l)] += theta * score_diff / sg_norm * sg
    return False, x, w, Zd_score, feasible_score, u_w, x_cands


def compute_score(P, x, use_binary=True):
    t0 = time.time()
    score = 0
    unary_score = 0
    binary_score = 0
    for i, j in enumerate(x):
        Dij = P.D[i, j]
        score += Dij
        unary_score += Dij
        B_part = P.B.get(j)
        if use_binary:
            for k in P.adj_list[i]:
                l = x[k]
                if not P.self_discount and (i == j or k == l):
                    continue
                # We could also call just P.B.get(j, l), but this is
                # slightly faster
                Bjl = P.B.get_j(B_part, l)
                if Bjl:
                    score -= Bjl * P.g / 2.0
                    binary_score -= Bjl * P.g / 2.0
    """
    if use_binary:
        for i,j,k,l in P.squares:
            if x[i] == j and x[k] == l:
                score -= P.g / 2.0
                binary_score -= P.g / 2.0
    """
    #print " Feasible unary: {}, binary: {}, x: {}".format(unary_score,
    #                                                      binary_score, x)
    return score


def compute_infeasible_score(P, x, w, u_w, use_binary=True):
    t0 = time.time()
    dscore = 0
    unary_score = 0
    binary_score = 0
    for i, j in enumerate(x):
        Dij = P.D[i, j]
        dscore += Dij
        unary_score += Dij
    if use_binary:
        for (i, j) in w.w.iterkeys():
            B_part = P.B.get(j)
            for k, ls in w.w[(i, j)].iteritems():
                for l in ls.iterkeys():
                    if w.get(i, j, k, l):
                        #Bjl = P.B.get(j, l)
                        Bjl = P.B.get_j(B_part, l)
                        u_ijkl = u_w.get((i, j, k, l), 0)
                        dscore += 2 * u_ijkl - Bjl * P.g / 2.0
                        binary_score += 2 * u_ijkl - Bjl * P.g / 2.0
    #print " Dual unary: {}, binary: {}".format(unary_score, binary_score)
    return dscore


def get_v_munkres(i, j, w, u_w, P):
    if (i,j) not in w.w:
        return 0
    v = 0
    k_vals = {}
    l_vals = {}
    kl_costs = {}
    min_val = 1e10
    for _, _, k, l, _ in w.iter(fixed_ij=(i, j)):
        if k not in k_vals:
            k_vals[k] = len(k_vals)
        val = 2 * u_w.get((i, j, k, l), 0) - P.B.get(j, l) * P.g / 2.0
        if val < min_val:
            min_val = val
        if val < 0:
            if l not in l_vals:
                l_vals[l] = len(l_vals)
            kl_costs[(k, l)] = val
    cost_infeasible = 10000000000
    assignment_costs = np.ones((len(k_vals), len(l_vals))) * cost_infeasible
    for (k, l), val in kl_costs.iteritems():
        assignment_costs[k_vals[k], l_vals[l]] = val - min_val + 1
    k_inv_vals = {val: key for key, val in k_vals.iteritems()}
    l_inv_vals = {val: key for key, val in l_vals.iteritems()}
    # Run Munkres algorithm
    row_ind, col_ind = min_cost_matching(assignment_costs)
    assignments = zip(row_ind, col_ind)

    for ki, li in assignments:
        k = k_inv_vals[ki]
        l = l_inv_vals[li]
        val = assignment_costs[ki, li] + min_val - 1
        if val < 0:
            v += val
            w.set(i, j, k, l)
    return v


def get_v_greedy(i, j, w, u_w, P):
    """
    Ignore the one-to-one constraint and simply match every node to the best
    matching entity (multiple nodes from one graph might get mapped to the same
    entity).
    """
    if (i, j) not in w.w:
        return 0
    v = 0
    matches = {}
    for _, _, k, l, _ in w.iter(fixed_ij=(i, j)):
        if k not in matches:
            matches[k] = (0, None)
        val = 2 * u_w.get((i, j, k, l), 0) - P.B.get(j, l) * P.g / 2.0
        if val < matches[k][0]:
            matches[k] = (val, l)

    for k, (cost, l) in matches.iteritems():
        if cost < 0 and l is not None:
            v += cost
            w.set(i, j, k, l)
    return v


def multi_match_x(P, x_candidates):
    """
    Assign each item to a candidate item by solving a matching problem.

    Input:
        P -- Problem instance.
        x_candidates -- list of candidate tuple list:
                        item -> candidate idx -> (score, j)
    """
    # Initially assign each item to the first candidate match
    x = [x_candidates[i][0][1] for i in range(len(x_candidates))]
    # Then start solving the matching for one input graph at a time
    graph_idxs = P.graph2nodes.keys()
    np.random.shuffle(graph_idxs)
    for gidx in graph_idxs:
        items = P.graph2nodes[gidx]
        candidate_set = set()
        used_entities = set()
        uncertain_i = []
        # First check all items with only a single candidate match
        for i in items:
            assert len(x_candidates[i]) > 0, \
                "Zero candidates for item {}.".format(i)
            if len(x_candidates[i]) == 1:
                used_entities.add(x_candidates[i][0][1])
        # Then assign the remaining items to unused entities
        min_score = np.inf
        for i in items:
            if len(x_candidates[i]) > 1:
                n_unused_candidates = 0
                for score, j in x_candidates[i]:
                    if j not in used_entities:
                        candidate_set.add(j)
                        n_unused_candidates += 1
                        if score < min_score:
                            min_score = score
                assert n_unused_candidates > 0, \
                    "Zero unused candidates for item {}.".format(i)
                uncertain_i.append(i)
        # No need to solve the matching if all has been matched
        if len(uncertain_i) == 0:
            continue
        i_vals = {i: val for val, i in enumerate(uncertain_i)}
        j_vals = {j: val for val, j in enumerate(candidate_set)}
        cost_infeasible = 10000000000
        assignment_costs = np.ones((len(i_vals), len(candidate_set))) * \
                           cost_infeasible
        for i, xi_candidates in enumerate(x_candidates):
            if i in i_vals:
                i_val = i_vals[i]
                for score, j in xi_candidates:
                    if j not in used_entities:
                        assignment_costs[i_val, j_vals[j]] = score - \
                                                             min_score + 1
        i_inv_vals = {val: key for key, val in i_vals.iteritems()}
        j_inv_vals = {val: key for key, val in j_vals.iteritems()}
        # Run Munkres algorithm
        row_ind, col_ind = min_cost_matching(assignment_costs)
        assignments = zip(row_ind, col_ind)

        for i_val, j_val in assignments:
            i = i_inv_vals[i_val]
            j = j_inv_vals[j_val]
            x[i] = j
            assert j in P.candidate_matches[i], \
                "{} cannot be assigned to {}.\n{}".format(i, j, x)
    return x


def greedy_x(P, x_candidates):
    """
    Assign each item to a candidate item greedily.

    Input:
        P -- Problem instance.
        x_candidates -- list of candidate tuple list:
                        item -> candidate idx -> (score, j)
    """
    x = []
    for cands in x_candidates:
        min_cost = np.inf
        min_j = None
        for cost, j in cands:
            if cost < min_cost:
                min_cost = cost
                min_j = j
        x.append(min_j)
    return x


def solve_xw(P, u_w, use_binary=True, greedy=False):
    w = variables.W(P)
    # x's and their match-score pairs
    x_candidates = []
    candidate_set = set()
    min_x_score = np.inf
    for i, j_vals in enumerate(P.candidate_matches):
        assert len(j_vals) > 0, \
            "Each item should have at least one candidate match (e.g. " + \
            "itself). Item {} has zero.".format(i)
        fx_i = []
        v_part = 0
        for j in j_vals:
            if use_binary:
                if not greedy:
                    v_ij = get_v_munkres(i, j, w, u_w, P)
                else:
                    #v_ij = get_v_munkres(i, j, w, u_w, P)
                    v_ij = get_v_greedy(i, j, w, u_w, P)
            else:
                v_ij = 0
            v_part += v_ij
            x_score = P.D[i, j] + v_ij
            fx_i.append((x_score, j))
            candidate_set.add(j)
            if x_score < min_x_score:
                min_x_score = x_score
        x_candidates.append(fx_i)
    if not greedy:
        x = multi_match_x(P, x_candidates)
    else:
        #x = multi_match_x(P, x_candidates)
        x = greedy_x(P, x_candidates)
    # Update w (w_ijkl should be 0 if x_ij is zero)
    new_w = variables.W(P)
    for i, j in enumerate(x):
        w_part = w.w.get((i, j), {})
        for k, ls in w_part.iteritems():
            for l in ls.iterkeys():
                if w.get(i, j, k, l):
                    new_w.set(i, j, k, l)
    w = new_w
    return x, w, x_candidates
