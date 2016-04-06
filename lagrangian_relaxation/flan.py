"""
FLAN method for aligning multiple networks.
"""
import numpy as np
import time
from scipy.optimize import linear_sum_assignment
from scipy import sparse
import variables
import feasibility_heuristics as heuristics
import networkx as nx

# For an unknown reason, the duality gap goes sometimes slightly negative.
# Whether to give an assertion error in those cases or not (a warning will be
# printed in any case).
require_positive_gap = False

# Print iteration scores only at every kth iteration
print_interval = 50
#print_interval = 1


def LD_algorithm(P, max_iterations=500, max_alg_iterations=1, binary_B=False,
                 use_binary=True, cluster=False):
    """
    FLAN algorithm.

    Input:
        P -- problem object (see variables.py)
        max_iterations -- for how many iterations do we update the Lagrangian
                          multipliers
        max_alg_iterations -- how many times we run the algorithm (e.g. value 3
                              means we run it three times and update edges twice
                              in between)
        binary_B -- whether to binarizy the adjacency or use a weighted one. The
                    latter seems to work better
        use_binary -- whether to use the quadratic (binary) term in the
                      objective
        cluster -- whether to cluster the prediction (similar to what is done
                   with natalie). Has little effect.
    """
    prev_xs = []
    for ai in range(max_alg_iterations):
        if ai > 0:
            P = update_B(P, x, binary_B)
        print "\nLD algorithm iteration {}.".format(ai)
        res = LD_algorithm_iteration(P, max_iterations, use_binary, cluster)
        x = res['best_x']
        prev_xs.append(x)
        # If x hasn't changed, stop the iteration
        if len(prev_xs) >= 2:
            all_same = True
            prev_x = prev_xs[-2]
            for i in range(len(x)):
                if x[i] != prev_x[i]:
                    all_same = False
                    break
            if all_same:
                print "\nx didn't change, stopping algorithm iteration."
                break
    res['prev_xs'] = prev_xs
    return res


def update_B(P, x, binary_B):
    """
    Update adjacency matrix B based on the previous assignment x.

    Output:
        a new problem object with an updated B
    """
    new_B = sparse.dok_matrix(P.B.shape, dtype=np.int)
    for i, j in enumerate(x):
        for k in P.adj_list[i]:
            l = x[k]
            if binary_B:
                new_B[j, l] = 1
            else:
                new_B[j, l] += 1
    new_P = variables.Problem(
        P.f, P.D, P.g, P.gap_cost, P.N, P.A, P.candidate_matches,
        P.graph2nodes, P.groundtruth, B=new_B, self_discount=P.self_discount,
        max_entities=P.max_entities
    )
    return new_P


def LD_algorithm_iteration(P, max_iterations=500, use_binary=True,
                           cluster=False):
    u_x = np.ones(P.N) * 0
    u_w = {}
    l = -np.inf
    u = np.inf
    best_x = None
    theta = 0.1
    sg_N = 20
    sg_M = 10
    n = sg_N
    m = sg_M
    if P.groundtruth is not None:
        gt_score = compute_score(P, P.groundtruth, use_binary=use_binary)
        print "Groundtruth score: {}".format(gt_score)
    Zd_scores = []
    feasible_scores = []
    prev_duality_gap = np.inf
    n_equal_gaps = 0
    for t in range(0, max_iterations):
        has_converged, x, y, w, Zd_score, feasible_x, feasible_score, u_x, u_w = \
                LD_iteration(P, u_x, u_w, theta, use_binary=use_binary)
        Zd_scores.append(Zd_score)
        feasible_scores.append(feasible_score)

        duality_gap = feasible_score - Zd_score
        rel_dg_change = np.abs(prev_duality_gap - duality_gap) / \
                        max(np.abs(feasible_score), np.abs(Zd_score))
        #rel_dg_change = np.abs(duality_gap) / max(np.abs(feasible_score),
        #                                          np.abs(Zd_score))
        if t % print_interval == 0:
            print "Duality gap {}, rel change {}".format(duality_gap, rel_dg_change)
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
        if feasible_score < u:
            u = feasible_score
            best_x = feasible_x
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
            print "{}: Zd score: {},\tfeasible score: {}".format(t, Zd_score,
                                                                 feasible_score)
        if require_positive_gap:
            assert Zd_score - feasible_score < 1e-4, Zd_score - feasible_score
        if has_converged:
            print "*** Iteration converged in {} steps! ***".format(t+1)
            break
    print "Best feasible score {}, lb: {}, ub: {}".format(u, l, u)
    #print "Best x: {}".format(best_x)
    if cluster:
        best_x = cluster_x(best_x, P.candidate_matches)
    results = {'best_x': best_x, 'Zd_scores': Zd_scores,
               'feasible_scores': feasible_scores, 'iterations': t+1,
               'lb': l, 'ub': u}
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


def LD_iteration(P, u_x, u_w, theta, use_binary=True):
    """
    Input:
        u_x -- lambdas for the \sum_j x_ij = 1 constraint
        u_w -- lambdas for the w_ijkl = w_klij constraint

    Output:
        Optimal solution (x,y,w) and Lagrange multipliers
    """
    x, y, w, feasible_x, feasible_y = solve_xyw(P, u_x, u_w,
                                                use_binary=use_binary)
    feasible_score = compute_score(P, feasible_x, use_binary=use_binary)
    Z_score, Zd_score = compute_infeasible_score(P, x, w, u_x, u_w,
                                                 use_binary=use_binary)
    # Duality gap
    score_diff = feasible_score - Zd_score
    #score_diff = max(feasible_score - Zd_score,0)
    if score_diff < -1e-5:
        print "\nWarning!!! Negative duality gap: {}".format(score_diff)
        #print "x: {}\nfx: {}\n".format(x[:100], feasible_x[:100])

    # Compute subgradient vectors
    sgx = []
    for i, x_js in enumerate(x):
        sgx.append(1 - len(x_js))
    sgw = []
    for i, j, k, l, val in w.iter():
        sgw.append(val - w.get(k, l, i, j))
    sgx_norm = np.linalg.norm(sgx, 2) ** 2
    sgw_norm = np.linalg.norm(sgw, 2) ** 2
    sg_norm = np.linalg.norm(sgx+sgw, 2) ** 2
    #print "sg norm:", sg_norm, sgx_norm, sgw_norm
    if sg_norm < 1e-5: # Converged
        return True, x, y, w, Zd_score, feasible_x, feasible_score, u_x, u_w

    # Update Lagrange multipliers
    for i, x_js in enumerate(x):
        sg = sgx[i]
        if sgx_norm > 0:
            u_x[i] += theta * score_diff / sg_norm * sg
    for i, j, k, l, val in w.iter():
        if (i, j, k, l) not in u_w:
            u_w[(i, j, k, l)] = 0
        sg = val - w.get(k, l, i, j)
        if sgw_norm > 0:
            u_w[(i, j, k, l)] += theta * score_diff / sg_norm * sg
    return False, x, y, w, Zd_score, feasible_x, feasible_score, u_x, u_w


def compute_score(P, x, use_binary=True):
    t0 = time.time()
    n_opened = len(set(x))
    if P.fix_max_entities:
        score = 0
    else:
        score = P.f * n_opened
    unary_score = 0
    binary_score = 0
    for i, j in enumerate(x):
        score += P.D[i, j]
        unary_score += P.D[i, j]
        if use_binary:
            for k in P.adj_list[i]:
                l = x[k]
                if P.B[j, l]:
                    score -= P.B[j, l] * P.g / 2.0
                    binary_score -= P.B[j, l] * P.g / 2.0
    """
    # An alternative way of computing the quadratic term
    if use_binary:
        for i,j,k,l in P.squares:
            if x[i] == j and x[k] == l:
                score -= P.B[j,l] * P.g / 2.0
                binary_score -= P.B[j,l] * P.g / 2.0
    """
    #print "F:   {} opened entities, {} unary, {} binary, score {} (took {} " +\
    #      "seconds).".format(n_opened, unary_score, binary_score, score,
    #                         time.time()-t0)
    return score


def compute_infeasible_score(P, x, w, u_x, u_w, use_binary=True):
    t0 = time.time()
    opened_y = set()
    for xi in x:
        opened_y |= set(xi)
    n_opened = len(opened_y)
    if P.fix_max_entities:
        score = 0
        dscore = np.sum(u_x)
    else:
        score = P.f * n_opened
        dscore = P.f * n_opened + np.sum(u_x)
    unary_score = 0
    binary_score = 0
    binary_score2 = 0
    for i, xi in enumerate(x):
        for j in xi:
            score += P.D[i, j]
            dscore += P.D[i, j] - u_x[i]
            unary_score += P.D[i, j] - u_x[i]
            if use_binary:
                for _, _, k, l in w.iter_nonzero(ij=(i, j)):
                    score -= P.B[j, l] * P.g / 2.0
                    dscore += 2*u_w.get((i, j, k, l),0) - P.B[j, l] * P.g / 2.0
                    binary_score += 2*u_w.get((i, j, k, l), 0) - P.B[j, l] * \
                                                                 P.g / 2.0
    return score, dscore


def get_v_munkres(i, j, w, u_w, P):
    if (i, j) not in w.w:
        return 0
    k_vals = {}
    l_vals = {}
    kl_costs = {}
    min_val = 1e10
    for _, _, k, l, _ in w.iter(fixed_ij=(i, j)):
        if k not in k_vals:
            k_vals[k] = len(k_vals)
        val = 2 * u_w.get((i, j, k, l), 0) - P.B[j, l] * P.g / 2.0
        if val < min_val:
            min_val = val
        if val < 0:
            if l not in l_vals:
                l_vals[l] = len(l_vals)
            kl_costs[(k, l)] = val
    cost_infeasible = 1000000000
    assignment_costs = np.ones((len(k_vals), len(l_vals))) * cost_infeasible
    for (k, l), val in kl_costs.iteritems():
        shifted_val = val - min_val + 1
        assignment_costs[k_vals[k], l_vals[l]] = shifted_val
        assert shifted_val < cost_infeasible / 10.0, \
                "Increase the value of cost_infeasible from {}".format(
                    shifted_val)
    k_inv_vals = {val: key for key, val in k_vals.iteritems()}
    l_inv_vals = {val: key for key, val in l_vals.iteritems()}
    # Run Munkres algorithm
    row_ind, col_ind = linear_sum_assignment(assignment_costs)
    assignments = zip(row_ind, col_ind)

    v = 0
    for ki, li in assignments:
        k = k_inv_vals[ki]
        l = l_inv_vals[li]
        val = assignment_costs[ki, li] + min_val -1
        if val < 0:
            v += val
            w.set(i, j, k, l)
    return v


def solve_xyw(P, u_x, u_w, use_binary=True):
    w = variables.W(P)
    x = []
    # x's and their match-score pairs
    x_candidates = []
    C = {}
    for i, j_vals in enumerate(P.candidate_matches):
        assert len(j_vals) > 0, "Each item should have at least one " + \
                                "candidate match (e.g. itself). Item " + \
                                "{} has zero.".format(i)
        x_i = []
        fx_i = []
        v_part = 0
        for j in j_vals:
            if use_binary:
                #v_ij = get_v(i, j, w, u_w, P.g)
                v_ij = get_v_munkres(i, j, w, u_w, P)
            else:
                v_ij = 0
            v_part += v_ij
            x_score = P.D[i, j] - u_x[i] + v_ij
            C[j] = C.get(j, 0) + min(0, x_score)
            if x_score < 0:
                x_i.append((x_score, j))
            fx_i.append((x_score, j))
        x.append(x_i)
        x_candidates.append(fx_i)
    # Open the optimal y's
    y = [0] * P.N
    if P.fix_max_entities:
        # y's and their scores
        y_scores = [0] * P.N
        for j, Cj in C.iteritems():
            y_scores[j] = Cj
        order = np.argsort(y_scores)
        for j in range(P.max_entities):
            y[order[j]] = 1
    else:
        # y's and their scores
        y_scores = [P.f] * P.N
        for j, Cj in C.iteritems():
            y_score = P.f + Cj
            #print "y_score[%d] = %f, C = %f" % (j, y_score, Cj)
            if y_score < 0:
                y_val = 1
            else:
                y_val = 0
            y[j] = y_val
            y_scores[j] = y_score
        #   print "{:.2f}\t{:.2f}\t{:.2f}".format(C, u_x[j], v_part)
        #print

    # Remove x's which shouldn't have been assigned
    new_x = []
    new_w = variables.W(P)
    for i in range(len(x)):
        new_x_i = []
        for x_score, j in x[i]:
            if y[j] == 1:
                new_x_i.append(j)
                # Keep track of how many are mapped to what to ensure a
                # feasible solution
                k_counts = {}
                l_counts = {}
                for _, _, k, l in w.iter_nonzero(ij=(i, j)):
                    k_counts[k] = k_counts.get(k, 0) + 1
                    assert k_counts[k] <= 1, \
                        "Value {} mapped to too many.".format(k)
                    l_counts[l] = l_counts.get(l, 0) + 1
                    assert l_counts[l] <= 1, \
                        "Value {} mapped to too many.".format(l)
                    new_w.set(i, j, k, l)
        new_x.append(new_x_i)
    x = new_x
    w = new_w

    # Use heuristics to make x feasible
    if P.fix_max_entities:
        feasible_x, feasible_y = heuristics.make_feasible_munkres_me(
            x_candidates, y, y_scores, P)
    else:
        feasible_x, feasible_y = heuristics.make_feasible_munkres(x_candidates, y_scores, P)
    """
    n_print = 50
    print " y: {}".format(y[:n_print])
    print "fy: {}".format(feasible_y[:n_print])
    print "fy2: {}".format(feasible_y2[:n_print])
    print " x: {}".format(x[:n_print])
    print "fx: {}".format(feasible_x[:n_print])
    print "fx2: {}".format(feasible_x2[:n_print])
    """
    return x, y, w, feasible_x, feasible_y
