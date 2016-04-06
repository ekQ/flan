"""
Run this file to conduct the experiment on social network alignment.
"""
import networkx as nx
import numpy as np
import random
import time
import datetime as dt
import os
import utilities as util

from align_multiple_networks import align_multiple_networks
import plot_experiment_results as plot


def read_multiplex_data(fname, n_duplicate_names=3, max_nodes=40,
                        plot_graphs=False):
    """
    Reads in the dataset from and edge file and creates a set of networks.
    Input:
        fname -- edge file path
        n_duplicate_names -- number of people who will be assigned the same name
        max_nodes -- maximum number of nodes per layer
    """
    E = np.loadtxt(fname, np.int)
    layers = E[:, 0].flatten()
    uniq_layers = list(set(layers))
    us = E[:, 1].flatten()
    vs = E[:, 2].flatten()
    ids = list(set(us) | set(vs))
    N = len(ids)
    print "{} ids and {} layers in total.".format(N, len(uniq_layers))

    compute_ids_per_layer(layers, E)

    # Generate names for the people
    n_uniq_names = int(np.ceil(N / float(n_duplicate_names)))
    name_pool = []
    for i in range(n_uniq_names):
        name_pool += [i] * n_duplicate_names
    names = {}
    for pid in ids:
        names[pid] = name_pool.pop(random.randint(0, len(name_pool)-1))

    # Create the networks
    Gs = []
    Gent = nx.Graph()
    print "n_uniq_names", n_uniq_names
    all_layer_ids = set()
    for ki, k in enumerate(uniq_layers):
        # Get ids on this layer
        idxs = layers == k
        layer_ids = list(set(us[idxs]) | set(vs[idxs]))
        if len(layer_ids) > max_nodes:
            layer_ids = random.sample(layer_ids, max_nodes)
        print "{} people on layer {}.".format(len(layer_ids), k)
        all_layer_ids |= set(layer_ids)
        G = nx.Graph()
        # Add nodes
        for pid in layer_ids:
            G.add_node(pid, name=names[pid], entity=pid, subgraph=k-1)
            Gent.add_node(pid, name=names[pid], entity=pid)
        Gs.append(G)
    print "{} people on layers together (after subsampling).".format(
        len(all_layer_ids))

    # Add edges
    assert len(us) == len(vs)
    edge_counts = [0] * len(uniq_layers)
    for i in range(len(us)):
        gidx = layers[i] - 1
        u = us[i]
        v = vs[i]
        if Gs[gidx].has_node(u) and Gs[gidx].has_node(v):
            Gs[gidx].add_edge(u, v)
            edge_counts[gidx] += 1
            Gent.add_edge(u, v)

    if plot_graphs:
        plot.plot_graphs(Gs+[Gent])

    # Order the graphs based on edge counts
    #order = np.argsort(edge_counts)[::-1]
    print "Edge counts:", edge_counts
    #print "Order:", order
    #Gs = [Gs[i] for i in order]
    return Gs


def compute_ids_per_layer(layers, E):
    # Uniq ids per layer
    id_set = set()
    for i in range(1, 6):
        ok_idxs = layers == i
        us = E[ok_idxs, 1].flatten()
        vs = E[ok_idxs, 2].flatten()
        ids = set(us) | set(vs)
        id_set |= ids
        N = len(ids)
        print "Layer {}: {} ids".format(i, N)
    print "Users in all layers:", len(id_set)


def multiplex_experiment(n_reps=10):
    """
    Run an experiment on alignining the (anonymized) layers of a multiplex
    graph.

    Input:
        n_reps -- number of repetitions per setting
    Output:
        Prints some statistics and stores the results to a file.
    """
    shuffle = True
    #methods = ['upProgmKlau']
    methods = ('ICM', 'progmKlau', 'upProgmKlau', 'mKlau', 'LD', 'LD5',
               'meLD5_61', 'meLD5_50', 'meLD5_70')
    g = 0.5
    max_iters = 300
    duplicate_names = 3
    f_values = [0.3, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5]
    nvv = len(f_values)

    fname = os.path.join('multiplex', 'CS-Aarhus_multiplex.edges')

    experiment_seed = np.random.randint(0, 1000000)
    # experiment_seed = 48574 # Gt yields a better optimum
    print "--- Experiment seed: {} ---\n".format(experiment_seed)
    random.seed(experiment_seed)
    np.random.seed(experiment_seed)

    res_precision = np.zeros((len(methods), nvv, n_reps))
    res_recall = np.zeros((len(methods), nvv, n_reps))
    res_fscore = np.zeros((len(methods), nvv, n_reps))
    res_iterations = np.zeros((len(methods), nvv, n_reps))
    res_t = np.zeros((len(methods), nvv, n_reps))
    res_clusters = np.zeros((len(methods), nvv, n_reps))
    res_costs = np.zeros((len(methods), nvv, n_reps))
    res_lb = np.zeros((len(methods), nvv, n_reps))  # Lower bounds
    res_ub = np.zeros((len(methods), nvv, n_reps))  # Upper bound

    t_beg = time.time()
    date0 = dt.datetime.now()
    for r in range(n_reps):
        print "\n  Repetition: {}".format(r)
        Gs = read_multiplex_data(fname, n_duplicate_names=duplicate_names)
        for i, f in enumerate(f_values):
            print "\nf={}.\n".format(f)
            cost_params = {'f': f, 'g': g, 'gap_cost': f}
            for j, method in enumerate(methods):
                print "\n  method={}, f={}, rep={}".format(method, f, r)
                max_entities = None
                mai = 1
                if method.startswith('LD') and len(method) > 2:
                    mai = int(method[2:])
                    method = 'LD'
                elif method.startswith('binB-LD') and len(method) > 7:
                    mai = int(method[7:])
                    method = 'binB-LD'
                elif method.startswith('meLD'):
                    parts = method.split('_')
                    if len(parts[0]) > 4:
                        mai = int(parts[0][4:])
                    max_entities = int(parts[1])
                    method = 'LD'
                t0 = time.time()
                x, o = align_multiple_networks(
                    Gs, cost_params, method=method, max_iters=max_iters,
                    max_algorithm_iterations=mai, max_entities=max_entities,
                    shuffle=shuffle)
                print "Optimization took {:.2f} seconds.".format(time.time() -
                                                                 t0)
                pr, rec, f1 = o['scores']

                res_t[j, i, r] = time.time() - t0
                res_precision[j, i, r] = pr
                res_recall[j, i, r] = rec
                res_fscore[j, i, r] = f1
                res_iterations[j, i, r] = o['iterations']
                res_clusters[j, i, r] = o['n_clusters']
                res_costs[j, i, r] = o['cost']
                res_lb[j, i, r] = o['lb']
                res_ub[j, i, r] = o['ub']
        fname0 = util.save_data(locals(), "multiplex", date0)
        print "Wrote the results of repetition {} to: {}\n".format(r+1, fname0)
    print "\nThe whole experiment took {:2f} seconds.".format(time.time() -
                                                              t_beg)
    fname = util.save_data(locals(), "multiplex")
    print "Wrote the results to: {}".format(fname)
    #plot_toy_experiment_results(fname)

    print "F1 score:", np.mean(res_fscore, axis=2)
    print "Precision:", np.mean(res_precision, axis=2)
    print "Recall:", np.mean(res_recall, axis=2)
    print "Time:", np.mean(res_t, axis=2)
    print "Iterations:", np.mean(res_iterations, axis=2)
    print "Clusters:", np.mean(res_clusters, axis=2)
    print "Costs:", np.mean(res_costs, axis=2)
    print "Lower bounds:", np.mean(res_lb, axis=2)
    print "Upper bounds:", np.mean(res_ub, axis=2)


def multiplex_single_test(method, n_duplicates=2):
    cost_params = {'f': 2, 'g': 0.5, 'gap_cost': 2}
    Gs = read_multiplex_data('multiplex/CS-Aarhus_multiplex.edges',
                             n_duplicate_names=n_duplicates)
    x, o = align_multiple_networks(
        Gs, cost_params, method=method, max_iters=300,
        max_algorithm_iterations=1, shuffle=True)
    print "Solution:", x
    pr, rec, f1 = o['scores']
    print "Precision:", pr
    print "Recall:", rec
    print "F-score:", f1


def main():
    seed = np.random.randint(0, 1000000)

    print "--- Seed: {} ---\n".format(seed)
    random.seed(seed)
    np.random.seed(seed)

    multiplex_experiment(n_reps=30)

if __name__ == "__main__":
    main()
