"""
Run this file to conduct the experiment on aligning multiple genealogical trees.
"""
from family_trees import person
import jellyfish
import time
import datetime as dt
import os
import numpy as np
import align_multiple_networks as amn
import cPickle as pickle
import networkx as nx
import re
from family_trees import extract_tree_fragment_dataset as extract_ft


def create_index(people):
    """
    Create an index data structure to efficiently retrieve people with similar
    characteristics.
    :param people: list of people
    :return: dict (blocking key -> list of matching people)
    """
    index = {}
    for p in people:
        x = p.bkey
        if x not in index:
            index[x] = [p]
        else:
            index[x].append(p)
    return index


def get_unaries(people, people2, index2, year_window, top_k_matches=10,
                people_dict=None, people2_dict=None, parent_sims=False):
    """
    For each Person in people return a list of potentially matching xrefs
    (in big_people) and another list containing the respective match
    probabilities.
    """
    unaries_xrefs = []
    unaries_probs = []
    for p in people:
        if p.bkey in index2:
            cands = index2[p.bkey]
        else:
            #print "No candidates for bkey:", p.bkey
            cands = people2
        matches = []
        for c in cands:
            sim = person_similarity(p, c, year_window, parent_sims, people_dict,
                                    people2_dict)
            matches.append((sim, c.xref))
        matches = sorted(matches, reverse=True)[:top_k_matches]
        xrefs = [tup[1] for tup in matches]
        unaries_xrefs.append(xrefs)
        #unaries_probs.append(np.ones(len(xrefs)) / float(len(xrefs)))
        unaries_probs.append([tup[0] for tup in matches])
    return unaries_xrefs, unaries_probs


def get_pairwise(xrefs1, xrefs2, big_people_dict, is_dad):
    pw = np.ones((len(xrefs1), len(xrefs2)))*0.5
    for i, xr1 in enumerate(xrefs1):
        for j, xr2 in enumerate(xrefs2):
            if is_dad:
                parent_xr = big_people_dict[xr1].dad
            else:
                parent_xr = big_people_dict[xr1].mom
            if parent_xr == xr2:
                pw[i, j] = 1
    return -np.log(pw)


def person_similarity(p1, p2, year_window, parent_sims=False, people_dict1=None,
                      people_dict2=None):
    # Don't match inaccurate and accurate dates (unrealistic)
    if (p1.byear is not None) != (p2.byear is not None):
        return 0
    # Don't match if dates are too different
    if (p1.byear is not None) and (p2.byear is not None) and \
                    abs(p1.byear-p2.byear) > year_window:
        return 0
    
    terms = 0
    sim_sum = 0
    
    sim_sum += jellyfish.jaro_winkler(p1.clean_first_name, p2.clean_first_name)
    terms += 1
    
    sim_sum += jellyfish.jaro_winkler(p1.clean_last_name, p2.clean_last_name)
    terms += 1
    
    sim = sim_sum / float(terms)
    
    if parent_sims and people_dict1 is not None and people_dict1 is not None:
        parent_factor = 0.5
        n_parents = 0
        recursive = False
        if p1.dad is not None and p2.dad is not None:
            sim += parent_factor * person_similarity(
                people_dict1[p1.dad], people_dict2[p2.dad], year_window,
                recursive, people_dict1, people_dict2)
            n_parents += 1

        if p1.mom is not None and p2.mom is not None:
            sim += parent_factor * person_similarity(
                people_dict1[p1.mom], people_dict2[p2.mom], year_window,
                recursive, people_dict1, people_dict2)
            n_parents += 1
        sim /= float(1 + parent_factor * n_parents)
    return sim


def merge_multiple(people_index_tuples, year_window, top_k_matches=5,
                   method='mKlau', uniq_people=None, f=1):
    """
    Merge multiple family trees.

    :param people_index_tuples: list of tuples (people, people_index)
    :param year_window: How different birth years are allowed to be.
    :param top_k_matches: Retrieve at most k candidate matches per person.
    :return: Predictions.
    """
    if method.endswith('++'):
        parent_sims = True
        method = method[:-2]
    else:
        parent_sims = False

    if method == 'unary':
        method = 'progmKlau'
        g = 0
        gap_cost = f
    else:
        g = 0.5
        gap_cost = f

    if method.startswith('meLD'):
        max_entities = uniq_people
        method = 'LD5'
    else:
        max_entities = None
    similarity_tuples = []
    n_multiple_matches = 0
    n_with_correct = 0
    for i, (people_i, index_i, people_dict_i) in enumerate(people_index_tuples):
        for j in range(i+1, len(people_index_tuples)):
            people_j, index_j, people_dict_j = people_index_tuples[j]
            xref2idx_j = {people_j[jj].xref: jj for jj in range(len(people_j))}
            unaries_xrefs, unaries_probs = get_unaries(
                    people_i, people_j, index_j, year_window,
                    top_k_matches=top_k_matches, people_dict=people_dict_i,
                    people2_dict=people_dict_j, parent_sims=parent_sims)
            for pi in range(len(unaries_xrefs)):
                xrefs = unaries_xrefs[pi]
                probs = unaries_probs[pi]
                for xref, prob in zip(xrefs, probs):
                    if prob < 0.8:
                        continue
                    pj = xref2idx_j[xref]
                    similarity_tuples.append((i, pi, j, pj, prob))
                    similarity_tuples.append((j, pj, i, pi, prob))
#                   print "{}, {} / {} -> {}, {} / {}: {}".format(
#                      i, pi, people_i[pi].xref, j, pj, people_j[pj].xref, prob)
                if len(xrefs) > 0:
                    n_multiple_matches += 1
                if people_i[pi].xref in xrefs:
                    n_with_correct += 1
        for pi in range(len(people_i)):
            similarity_tuples.append((i, pi, i, pi, 1.0))
    Gs = []
    for graph_idx, (people, _, _) in enumerate(people_index_tuples):
        xref2idx = {people[idx].xref: idx for idx in range(len(people))}
        G = nx.Graph()
        for i, p in enumerate(people):
            G.add_node(i, entity=p.xref)
            neighs = [p.dad, p.mom]
            is_dads = [True, False]
            for neigh, is_dad in zip(neighs, is_dads):
                if neigh is not None:
                    j = xref2idx[neigh]
                    G.add_edge(i, j)
        Gs.append(G)

    print "{} with multiple matches, {} including a correct match.".format(
        n_multiple_matches, n_with_correct
    )

    cost_params = {'f': f, 'g': g, 'gap_cost': gap_cost}
    # Do the inference
    t0 = time.time()
    print cost_params
    if method.startswith('LD') and len(method) > 2:
        mai = int(method[2:])
        method = 'LD'
    elif method.startswith('binB-LD') and len(method) > 7:
        mai = int(method[7:])
        method = 'binB-LD'
    else:
        mai = 1
    x, other = amn.align_multiple_networks(
            Gs, cost_params, similarity_tuples, method=method, max_iters=300,
            max_algorithm_iterations=mai, max_entities=max_entities,
            self_discount=True, shuffle=True)
    print "Inference took {0:.2f} seconds.".format(time.time() - t0)

    res = []
    G = other['G']
    for i, mi in enumerate(x):
        if i != mi:
#            print "{} to {}".format(G.node[i]['entity'], G.node[mi]['entity'])
            res.append(G.node[i]['entity'] == G.node[mi]['entity'])
    print "{} non-self predictions, accuracy {}.".format(
        len(res), float(np.sum(res)) / max(len(res), 1))

    precision, recall, fscore = other['scores']
    n_clusters = other['n_clusters']
    lb = other['lb']
    ub = other['ub']
    iters = other['iterations']
    return precision, recall, fscore, n_clusters, lb, ub, iters


def count_unique_people(tree_files):
    all_xrefs = set()
    for tf in tree_files:
        _, people_dict = person.read_people(tf, clean=True)
        all_xrefs |= set(people_dict.keys())
    print "{} unique xrefs.".format(len(all_xrefs))
    return len(all_xrefs)


def experiment_multiple_trees(n_reps=1, n_trees=5, n_people=500,
                              methods=('unary', 'LD', 'mKlau'),
                              top_k_matches=5, f_vals=(0.1, 0.5, 1, 1.5, 2)):
    nvv = len(f_vals)
    res_precision = np.zeros((len(methods), nvv, n_reps))
    res_recall = np.zeros((len(methods), nvv, n_reps))
    res_fscore = np.zeros((len(methods), nvv, n_reps))
    res_t = np.zeros((len(methods), nvv, n_reps))
    res_iterations = np.zeros((len(methods), nvv, n_reps))
    res_clusters = np.zeros((len(methods), nvv, n_reps))
    res_lb = np.zeros((len(methods), nvv, n_reps))
    res_ub = np.zeros((len(methods), nvv, n_reps))
    t_beg = time.time()

    start_date_part = str(dt.datetime.now())[:19]
    start_date_part = re.sub(' ', '_', start_date_part)
    start_date_part = re.sub(':', '', start_date_part)
    fname0 = os.path.join("experiment_results", "genealogical_{}.pckl".format(
        start_date_part))

    for j in range(n_reps):
        print "\n--- Repetition {}. ---".format(j+1)
        # Generate data
        tree_files = extract_ft.get_k_fragments(n_trees, n_people,
                                                label="first{}".format(j))
        people_index_tuples = []
        for tf in tree_files:
            people, people_dict = person.read_people(tf, clean=True)
            #'family_trees/data/rand_frag_%d/' % i, clean=True)
            index = create_index(people)
            people_index_tuples.append((people, index, people_dict))
        uniq_people = count_unique_people(tree_files)

        for i, f in enumerate(f_vals):
            print "\n  rep={}, f={}".format(j+1, f)
            for mi, m in enumerate(methods):
                print "\n    rep={}, f={}, method={}\n".format(j+1, f, m)
                t0 = time.time()
                precision, recall, fscore, n_clusters, lb, ub, iters = \
                    merge_multiple(people_index_tuples, 10, top_k_matches,
                                   method=m, uniq_people=uniq_people, f=f)
                res_precision[mi, i, j] = precision
                res_recall[mi, i, j] = recall
                res_fscore[mi, i, j] = fscore
                res_clusters[mi, i, j] = n_clusters
                res_t[mi, i, j] = time.time() - t0
                res_iterations[mi, i, j] = iters
                res_lb[mi, i, j] = lb
                res_ub[mi, i, j] = ub
        pickle.dump(locals(), open(fname0, 'wb'))
        print "Wrote the results of repetition {} to: {}\n".format(j+1, fname0)

    print "\nThe whole experiment took {:2f} seconds.".format(time.time()-t_beg)
    date_part = str(dt.datetime.now())[:19]
    date_part = re.sub(' ', '_', date_part)
    date_part = re.sub(':', '', date_part)
    fname = os.path.join("experiment_results", "genealogical_{}.pckl".format(
        date_part))
    pickle.dump(locals(), open(fname, 'wb'))
    print "Wrote the results to: {}\n".format(fname)

    print "F1 score:", np.mean(res_fscore, axis=2)
    print "Precision:", np.mean(res_precision, axis=2)
    print "Recall:", np.mean(res_recall, axis=2)
    print "Time:", np.mean(res_t, axis=2)
    print "Clusters:", np.mean(res_clusters, axis=2)
    print "Lower bounds:", np.mean(res_lb, axis=2)
    print "Upper bounds:", np.mean(res_ub, axis=2)


if __name__ == "__main__":
    methods = ['ICM', 'progmKlau', 'upProgmKlau', 'mKlau', 'LD', 'LD5', 'meLD',
               'unary', 'unary++', 'upProgmKlau++']
    f_vals = (0.1, 0.15, 0.2, 0.3, 0.5, 1, 1.5, 2, 3)
    experiment_multiple_trees(n_reps=10, n_trees=10, n_people=100,
                              methods=methods, top_k_matches=5, f_vals=f_vals)