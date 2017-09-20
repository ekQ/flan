"""
Generic utilities.
"""
import time
import datetime as dt
import re
import os
import cPickle as pickle


def save_data(data, title='saved_data', date=None, dir_name=None):
    date_part = get_date_str(date)
    dir_path = 'experiment_results'
    if dir_name is not None:
        dir_path = os.path.join(dir_path, dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    fname = os.path.join(dir_path, "{}_{}.pckl".format(title, date_part))
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
        assert P.A.get(item, neigh) == 1, "Non-neighbor in neighbor list"
        c -= P.g * (P.A.get(dst_clust, neigh_clust) -
                    P.A.get(cur_clust, neigh_clust))
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
            c += P.g * (P.A.get(cur_clust, other_clust) -
                        P.A.get(dst_clust, other_clust))
        elif P.A.get(dst_clust, other_clust):
            # Two non-neighs (from the same input graph) mapped to neighs
            c += 0  #P.g# / 5.0
    return c


def write_problem_to_file(Gs, sim_tups, e_fname, s_fname):
    with open(e_fname, 'w') as fe:
        for gidx, G in enumerate(Gs):
            for u, v in G.edges():
                fe.write("G{} {} {}\n".format(gidx, u, v))
        print "Wrote edges to:", e_fname

    with open(s_fname, 'w') as fs:
        for tup in sim_tups:
            fs.write("G{} {} G{} {} {}\n".format(*tup))
        print "Wrote candidate matches to:", s_fname
