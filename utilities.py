"""
Generic utilities.
"""
import time
import datetime as dt
import re
import os
import cPickle as pickle


def save_data(data, title='saved_data', date=None):
    date_part = get_date_str(date)
    fname = os.path.join("experiment_results", "{}_{}.pckl".format(title,
                                                                   date_part))
    pickle.dump(data, open(fname, 'wb'))
    print "Wrote the results to: {}".format(fname)
    return fname


def get_date_str(date=None):
    if date is None:
        date = dt.datetime.now()
    date_part = str(date)[:19]
    date_part = re.sub(' ', '_', date_part)
    date_part = re.sub(':', '', date_part)
    return date_part


def cost(assignment, P):
    t0 = time.time()
    X, x = assignment.construct_Xx()
    sum_y = assignment.count_opened_entities()
    unary_sum = (P.D.multiply(X)).sum()
    if P.AA.shape[0] == x.shape[0]:
        binary_sum = P.g * (x.T.dot(P.AA).dot(x))[0, 0] / 2.0
    elif P.A.shape[0] == X.shape[0]:
        qterm = P.A.dot(X).dot(P.A).tolil()
        qterm = qterm.reshape((x.shape[0], 1))
        binary_sum = P.g * (x.T.dot(qterm))[0, 0] / 2.0
    else:
        assert 1==0
    print P.f, sum_y, unary_sum, -binary_sum
    c = P.f*sum_y + unary_sum - binary_sum
    print "Cost took %f seconds." % (time.time()-t0)
    return c


def cost_BROKEN(assignment, P):
    t0 = time.time()
    X, x = assignment.construct_Xx()
    print "Xx", X.shape, x.shape
    sum_y = assignment.count_opened_entities()
    unary_sum = (P.D.multiply(X)).sum()
    assert P.A.shape[0] == X.shape[0]
    qterm = P.A.dot(X).dot(P.A).tolil()
    print x.shape, qterm.shape
    qterm = qterm.reshape((x.shape[0], 1))
    binary_sum = P.g * (x.T.dot(qterm))[0, 0] / 2.0
    print P.f, sum_y, unary_sum, -binary_sum
    c = P.f*sum_y + unary_sum - binary_sum
    print "Cost took %f seconds." % (time.time()-t0)
    return c


def delta_cost(item, dst_clust, assignment, P):
    cur_clust = assignment.matches[item]
    if cur_clust == dst_clust:
        return 0
    c = 0
    if len(assignment.clusters.get(cur_clust, [])) == 1:
        # Emptying cluster
        c -= P.f
    if len(assignment.clusters.get(dst_clust, [])) == 0:
        c += P.f

    # Distance cost
    c += P.D[item,dst_clust] - P.D[item, cur_clust]

    # Binary costs
    for neigh in P.adj_list[item]:
        neigh_clust = assignment.matches[neigh]
        assert P.A[item, neigh] == 1, "Non-neighbor in neighbor list"
        c -= P.g * (P.A[dst_clust, neigh_clust] - P.A[cur_clust, neigh_clust])
    return c


def delta_cost2(item, dst_clust, assignment, P):
    cur_clust = assignment.matches[item]
    if cur_clust == dst_clust:
        return 0
    c = 0
    if len(assignment.clusters.get(cur_clust, [])) == 1:
        # Emptying cluster
        c -= P.f
    if len(assignment.clusters.get(dst_clust, [])) == 0:
        c += P.f

    # Distance cost
    c += P.D[item,dst_clust] - P.D[item, cur_clust]

    # Binary costs
    for other in P.graph2nodes[P.node2graph[item]]:
        if other == item:
            continue
        other_clust = assignment.matches[other]
        if dst_clust == other_clust:
            # Two distinct mapped to the same
            # TODO Should this really cost g and not more (g seems to work
            # better)
            c += P.g
        if other in P.adj_list[item]:
            # Two neighs mapped to non-neighs OR neighs
            c += P.g * (P.A[cur_clust, other_clust] -
                        P.A[dst_clust, other_clust])
        elif P.A[dst_clust, other_clust]:
            # Two non-neighs (from the same input graph) mapped to neighs
            c += 0  #P.g# / 5.0
    return c
