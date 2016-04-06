"""
Heuristics to extract a feasible solution out of an infeasible one.

Requires: SciPy v0.17 or newer.
"""

import numpy as np
import random
from scipy.optimize import linear_sum_assignment


def make_feasible_munkres(x_candidates, y_scores, P):
    """
    Feasibility heuristics for (uncapacitated) FLAN.

    Input:
        x_candidates -- (list of lists of 2-tuples) Node i -> jth candidate
                        entity -> (score, entity j)
        y_scores -- (list) costs for opening different entities
        P -- Problem object
    """
    new_x = [i for i in range(P.N)]
    new_y = [0] * P.N
    y_scores2 = y_scores[:]

    graph_idxs = P.graph2nodes.keys()
    random.shuffle(graph_idxs)

    # First assign all items with only a single candidate to that candidate
    for i, cands in enumerate(x_candidates):
        assert len(cands) > 0, "Zero candidates for item {}".format(i)
        if len(cands) == 1:
            j = cands[0][1]
            new_x[i] = j
            new_y[j] = 1
            # Give a very small y_score to the opened entity so that it will
            # be preferred over closed entities in the matching
            y_scores2[j] = -1e8
    # Solve the bipartite matching for the remaining items one graph at a time
    for gidx in graph_idxs:
        items = P.graph2nodes[gidx]
        # Find (again) the items with only one candidate so that that candidate
        # is not considered as a potential entity for the other items in the
        # same graph
        used_entities = set()
        for i in items:
            assert len(x_candidates[i]) > 0, "Zero candidates for item {}".\
                format(i)
            if len(x_candidates[i]) == 1:
                j = x_candidates[i][0][1]
                used_entities.add(j)
        # Then formulate the assignment problem for the ambiguous items
        candidate_set = set()
        unassigned_items = []
        for i in items:
            i_cands = x_candidates[i]
            if len(i_cands) <= 1:
                continue
            unassigned_items.append(i)
            for cand_score, cand in i_cands:
                if cand not in used_entities:
                    candidate_set.add(cand)
        if len(unassigned_items) == 0:
            continue
        i_vals = {i: val for val, i in enumerate(unassigned_items)}
        j_vals = {j: val for val, j in enumerate(candidate_set)}
        cost_infeasible = 1e10
        assignment_costs = np.ones((len(i_vals), len(candidate_set))) * \
                           cost_infeasible
        for i in unassigned_items:
            i_val = i_vals[i]
            xi_candidates = x_candidates[i]
            for score, j in xi_candidates:
                y_score = y_scores2[j]
                if y_score < 0:
                    # We should assign to an entity with negative y_score if
                    # possible
                    y_score = -1e8
                assignment_costs[i_val, j_vals[j]] = score + y_score
        i_inv_vals = {val: key for key, val in i_vals.iteritems()}
        j_inv_vals = {val: key for key, val in j_vals.iteritems()}
        # Run Munkres algorithm
        row_ind, col_ind = linear_sum_assignment(assignment_costs)
        assignments = zip(row_ind, col_ind)

        for i_val, j_val in assignments:
            i = i_inv_vals[i_val]
            j = j_inv_vals[j_val]
            new_x[i] = j
            new_y[j] = 1
            y_scores2[j] = -1e8
    # Make sure that the solution is feasible
    for i, j in enumerate(new_x):
        assert j in P.candidate_matches[i], "{} cannot be assigned to {}.\n{}".\
            format(i, j, P.candidate_matches[i])
    return new_x, new_y


def make_feasible_munkres_me(x_candidates, y, y_scores, P):
    """
    Munkres algorithm based feasibility heuristics for the case of fixed number
    of maximum entities.

    Input:
        x_candidates -- (list of lists of 2-tuples) Node i -> jth candidate
                        entity -> (score, entity j)
        y_scores -- (list) costs for opening different entities
        P -- Problem object
    """
    y = open_max_entities(P, y_scores, x_candidates)

    new_x = [i for i in range(P.N)]
    new_y = [0] * P.N

    graph_idxs = P.graph2nodes.keys()
    random.shuffle(graph_idxs)

    # First assign all items with only a single candidate to that candidate
    for i, cands in enumerate(x_candidates):
        assert len(cands) > 0, "Zero candidates for item {}".format(i)
        if len(cands) == 1:
            j = cands[0][1]
            new_x[i] = j
            new_y[j] = 1
            assert y[j] == 1, "y[j] should be open."
            #print "Assigned {} to {}.".format(i, j)
    # Solve the bipartite matching for the remaining items one graph at a time
    for gidx in graph_idxs:
        items = P.graph2nodes[gidx]
        # Find (again) the items with only one candidate so that that candidate
        # is not considered as a potential entity for the other items in the
        # same graph
        used_entities = set()
        for i in items:
            assert len(x_candidates[i]) > 0, "Zero candidates for " + \
                                             "item {}".format(i)
            if len(x_candidates[i]) == 1:
                j = x_candidates[i][0][1]
                used_entities.add(j)
        # Then formulate the assignment problem for the ambiguous items
        candidate_set = set()
        unassigned_items = []
        for i in items:
            i_cands = x_candidates[i]
            if len(i_cands) <= 1:
                continue
            unassigned_items.append(i)
            for cand_score, cand in i_cands:
                if cand not in used_entities:
                    candidate_set.add(cand)
        if len(unassigned_items) == 0:
            continue
        i_vals = {i: val for val, i in enumerate(unassigned_items)}
        j_vals = {j: val for val, j in enumerate(candidate_set)}
        cost_infeasible = 1e10
        assignment_costs = np.ones((len(i_vals), len(candidate_set))) * \
                           cost_infeasible
        for i in unassigned_items:
            i_val = i_vals[i]
            xi_candidates = x_candidates[i]
            for score, j in xi_candidates:
                y_score = 0
                if y[j] == 0:
                    # We should have no need to assign to a closed entity so
                    # let's give it a very high cost
                    y_score += cost_infeasible / 100.0
                assignment_costs[i_val, j_vals[j]] = score + y_score
        i_inv_vals = {val: key for key, val in i_vals.iteritems()}
        j_inv_vals = {val: key for key, val in j_vals.iteritems()}
        # Run Munkres algorithm
        row_ind, col_ind = linear_sum_assignment(assignment_costs)
        assignments = zip(row_ind, col_ind)

        for i_val, j_val in assignments:
            i = i_inv_vals[i_val]
            j = j_inv_vals[j_val]
            new_x[i] = j
            new_y[j] = 1
            if y[j] == 0:
                print "Assigned {} ({}) to a closed entity {} ({}) with " + \
                      "cost {}.".format(i_val, i, j_val, j,
                                        assignment_costs[i_val, j_val])
                print assignment_costs
                assert False
    # Make sure that the solution is feasible
    for i, j in enumerate(new_x):
        assert j in P.candidate_matches[i], "{} cannot be assigned to {}.\n{}".\
            format(i, j, P.candidate_matches[i])#new_x)

    n_opened = sum(new_y)
    #if n_opened > P.max_entities:
    #print "\nEnded up opening {} entities instead of {}.\n".format(
    #    n_opened, P.max_entities
    #)
    return new_x, new_y


def open_max_entities(P, y_scores, x_candidates):
    # Collect all nodes that can be aligned to a given y
    y2cands = {}
    for xi, xi_candidates in enumerate(x_candidates):
        xi_gidx = P.node2graph[xi]
        for _, j in xi_candidates:
            if j not in y2cands:
                y2cands[j] = {}
            if xi_gidx not in y2cands[j]:
                y2cands[j][xi_gidx] = set()
            y2cands[j][xi_gidx].add(xi)
    # Start opening entities greedily starting from the one with most graphs and
    # best y_scores
    unaligned_xs = set(range(len(x_candidates)))
    opened_y = set()
    i = 0
    while len(unaligned_xs) > 0:
        # Pick greedily an y to open
        cand_g_counts = [(len(cgs), -y_scores[yi], yi) for yi, cgs in
                         y2cands.iteritems()]
        max_cgc = max(cand_g_counts)
        _, _, chosen_y = max_cgc
        cand_g_counts.remove(max_cgc)
        opened_y.add(chosen_y)
        # Align nodes to this y
        parts_to_remove = []
        for gidx, xis in y2cands[chosen_y].iteritems():
            # TODO: Pick x with most unopened candidates instead of a random x
            aligned_x = list(xis)[0]
            parts_to_remove.append((chosen_y, gidx))
            unaligned_xs.remove(aligned_x)
            # Remove aligned x from y2cands
            for other_y, y2cands_part in y2cands.iteritems():
                if other_y == chosen_y:
                    continue
                if gidx in y2cands_part:
                    y2cands_part[gidx].discard(aligned_x)
                    if len(y2cands_part[gidx]) == 0:
                        parts_to_remove.append((other_y, gidx))
        for other_y, other_gidx in parts_to_remove:
            y2cands[other_y].pop(other_gidx)
        i += 1
        if i > 1000000:
            print "Unable to open y's so that all x's can be aligned"
            print opened_y
            break
    min_opened = len(opened_y)
    if min_opened > P.max_entities:
        print "Had to open {} although the maximum was set to {}.".format(
            min_opened, P.max_entities
        )
    # Open additional y's if max_entities not reached yet
    y_tuples = sorted([(score, yi) for yi, score in enumerate(y_scores)])
    if min_opened < P.max_entities:
        for _, yi in y_tuples:
            if yi not in opened_y:
                opened_y.add(yi)
                if len(opened_y) >= P.max_entities:
                    break
    new_y = [0] * len(y_scores)
    for yi in opened_y:
        new_y[yi] = 1
    return new_y


def make_feasible(x_candidates, y, y_scores, P, u_x):
    """
    Heuristics to make x feasible.
    """
    # Set of x's for which there is at least one y opened
    covered_x = set()
    feasible_y = []
    for j, val in enumerate(y):
        feasible_y.append(val)
        if val == 1:
            covered_x |= P.candidates_to_match[j]
    # Open new y's if needed
    if len(covered_x) < P.N:
        y_scores = sorted(y_scores)
        for j, y_score in enumerate(y_scores):
            ctm = P.candidates_to_match[j]
            # Check if y[j] is closed and opening it would help
            if feasible_y[j] == 0 and len(ctm - covered_x) > 0:
                feasible_y[j] = 1
                covered_x |= ctm
                if len(covered_x) == P.N:
                    break
    feasible_x = []
    graph2matches = {}
    score = np.sum(u_x)
    for i in range(len(x_candidates)):
        best_x_j = None
        best_x_score = np.inf
        for x_score, j in x_candidates[i]:
            #print "Making feasible:", j, i, x_score
            if feasible_y[j] == 1 and j not in graph2matches.get(
                    P.node2graph[i], []):
                if x_score < best_x_score:
                    best_x_score = x_score
                    best_x_j = j
        if best_x_j is None:
            print "No best_x_j for {}.".format(i)
            if i not in graph2matches.get(P.node2graph[i],[]):
                feasible_y[i] = 1
                best_x_j = i
            else:
                print "X[{}] was not assigned.".format(i)
                print x_candidates[i]
                print graph2matches
                print P.node2graph[i]
                print feasible_x
                assert False
        feasible_x.append(best_x_j)
        score += best_x_score
        graph_idx = P.node2graph[i]
        if graph_idx not in graph2matches:
            graph2matches[graph_idx] = set()
        graph2matches[graph_idx].add(best_x_j)
    score += P.f * len(set(feasible_y))
    assert len(feasible_x) == len(feasible_y)
    return feasible_x, feasible_y
