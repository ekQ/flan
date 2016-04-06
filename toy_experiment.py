"""
Run this file to conduct the experiment on aligning synthetic graphs.
"""
from generate_toy_data import generate_graphs
from align_multiple_networks import align_multiple_networks
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import datetime as dt
import sys
from plot_experiment_results import plot_toy_experiment_results
import utilities as util


def cross_validate_params(method, n_input_graphs, n_entities,
                          n_input_graph_nodes, p_keep_edge, density_multiplier,
                          n_duplicates, max_iters, max_algorithm_iterations):
    """
    Return optimal parameters based on cross-validation.
    """
    print "\nStarting cross validation."
    sGs, G = generate_graphs(n_entities, n_input_graphs, n_input_graph_nodes,
                             p_keep_edge, density_multiplier, n_duplicates)
    fs = [0.1, 0.3, 0.48, 0.5, 0.52, 0.75, 1, 1.25, 1.5, 2, 4, 6, 8]
    g = 0.5
    scores = []
    for i, f in enumerate(fs):
        print "\nf = {}\n".format(f)
        cost_params = {'f': f, 'g': g, 'gap_cost': f}
        x, other = align_multiple_networks(
            sGs, cost_params, method=method, max_iters=max_iters,
            max_algorithm_iterations=max_algorithm_iterations,
            shuffle=True)
        precision, recall, fscore = other['scores']
        scores.append(fscore)
    print "\n\nOptimal scores: {}\n\n".format(scores)
    best_idx = np.argmax(scores)
    optimal_params = {'f': fs[best_idx], 'g': g, 'gap_cost': fs[best_idx]}
    return optimal_params


def single_cer(f=0.4, g=0.5, gap_cost=2, seed=None, method="ICM",
               n_input_graphs=3, n_entities=50, n_input_graph_nodes=50,
               p_keep_edge=0.8, density_multiplier=1.0, n_duplicates=5,
               max_iters=100, max_algorithm_iterations=1, shuffle=True,
               max_entities=None):
    """
    Run a single synthetic multiple network alignment test.
    """
    if seed is not None:
        print "Used seed:", seed
        random.seed(seed)
        np.random.seed(seed)
    sGs, G = generate_graphs(n_entities, n_input_graphs, n_input_graph_nodes,
                             p_keep_edge, density_multiplier, n_duplicates)
    t0 = time.time()
    cost_params = {'f': f, 'g': g, 'gap_cost': gap_cost}
    x, other = align_multiple_networks(
        sGs, cost_params, method=method, max_iters=max_iters,
        max_algorithm_iterations=max_algorithm_iterations,
        max_entities=max_entities, shuffle=shuffle,
    )
    print "Optimization took {:.2f} seconds.".format(time.time() - t0)
    precision, recall, fscore = other['scores']
    return precision, recall, fscore, other


def full_n_graphs(n_reps=3, full=False):
    """
    Test varying the number of input graphs.
    """
    varied_param = 'n_input_graphs'
    p = {}
    p[varied_param] = [2, 3, 4, 5, 6, 7]
    p['duplicates'] = 5
    p['density_multiplier'] = 1.5
    p['p_keep_edge'] = 0.7
    p['g'] = 0.5
    p['f'] = 2
    p['gap_cost'] = p['f']
    p['n_entities'] = 50
    if full:
        p['n_input_graph_nodes'] = 50
    else:
        p['n_input_graph_nodes'] = 30
    p['max_iters'] = 500
    full_str = 'full'
    cv = False
    if not full:
        full_str = 'partial'
        cv = True
    experiment_template(n_reps, p, varied_param, cv=cv,
                        title=full_str + '_n_graphs')


def f_effect(n_reps=10):
    """
    Test varying parameter f.
    """
    p = {}
    p['n_input_graphs'] = 10
    p['duplicates'] = 3
    p['density_multiplier'] = 1.1
    p['p_keep_edge'] = 0.8
    p['g'] = 0.5
    p['f'] = 0.4
    p['gap_cost'] = p['f']
    p['n_entities'] = 100
    p['n_input_graph_nodes'] = 30
    p['max_iters'] = 300

    varied_param = 'f'
    p[varied_param] = list(np.arange(0.05, 0.61, 0.05)) + \
                      [0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.5]
    methods = ['ICM', 'progmKlau', 'upProgmKlau', 'mKlau', 'LD', 'LD5',
               'meLD5_75', 'meLD5_100', 'meLD5_125']

    experiment_template(n_reps, p, varied_param, cv=False, methods=methods,
                        title='f_effect')


def max_entities_effect(n_reps=10, e_seed=None):
    """
    Test varying the number of entities.
    """
    p = {}
    p['n_input_graphs'] = 10
    p['duplicates'] = 3
    p['density_multiplier'] = 1.1
    p['p_keep_edge'] = 0.8
    p['g'] = 0.5
    p['f'] = 0.4
    p['gap_cost'] = p['f']
    p['n_entities'] = 100
    p['n_input_graph_nodes'] = 30
    p['max_iters'] = 300

    varied_param = 'f'
    p[varied_param] = [0.4]
    max_entities = [75, 100, 125]
    methods = ['meLD5_{}'.format(me) for me in max_entities]

    experiment_template(n_reps, p, varied_param, cv=False, methods=methods,
                        title='max_entities', e_seed=e_seed)


def vary_rem_noise(n_reps=10):
    """
    Test varying the edge removal probability.
    """
    p = {}
    p['n_input_graphs'] = 4
    p['duplicates'] = 5
    p['density_multiplier'] = 1
    p['p_keep_edge'] = 0.05
    p['g'] = 0.5
    p['f'] = 2
    p['gap_cost'] = p['f']
    p['n_entities'] = 50
    p['n_input_graph_nodes'] = 50
    p['max_iters'] = 500

    varied_param = 'p_keep_edge'
    p[varied_param] = np.arange(0, 1.01, 0.1)

    experiment_template(n_reps, p, varied_param, cv=False, title='rem_noise')


def vary_add_noise(n_reps=10):
    """
    Test varying the edge adding probability.
    """
    p = {}
    p['n_input_graphs'] = 4
    p['duplicates'] = 5
    p['density_multiplier'] = 1
    p['p_keep_edge'] = 0.8
    p['g'] = 0.5
    p['f'] = 2
    p['gap_cost'] = p['f']
    p['n_entities'] = 50
    p['n_input_graph_nodes'] = 50
    p['max_iters'] = 500

    varied_param = 'density_multiplier'
    p[varied_param] = [1, 1.1, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3]

    experiment_template(n_reps, p, varied_param, cv=False, title='add_noise')


def experiment_template(
        n_reps, params, varied_param, cv=False,
        methods=('ICM', 'progmKlau', 'upProgmKlau', 'mKlau', 'LD', 'LD5'),
        title='generic', e_seed=None):
    """
    General template for performing experiments.
    
    Input:
        n_reps -- number of repetitions per setting
        params -- all parameters (the parameter to be varied should be a list)
        varied_param -- the name of the parameter to be varied
        cv -- whether to find f and gap_cost through cross-validation

    Output:
        Prints some statistics and stores the results to a file.
    """
    shuffle = True

    if e_seed is None:
        experiment_seed = np.random.randint(0, 1000000)
    else:
        experiment_seed = e_seed
    # experiment_seed = 48574 # Gt yields a better optimum
    print "--- Experiment seed: {} ---\n".format(experiment_seed)
    random.seed(experiment_seed)
    np.random.seed(experiment_seed)

    varied_values = params[varied_param]
    nvv = len(varied_values)
    p = dict(params)  # Current values

    if 'max_entities' not in p:
        p['max_entities'] = None

    res_precision = np.zeros((len(methods), nvv, n_reps))
    res_recall = np.zeros((len(methods), nvv, n_reps))
    res_fscore = np.zeros((len(methods), nvv, n_reps))
    res_iterations = np.zeros((len(methods), nvv, n_reps))
    res_t = np.zeros((len(methods), nvv, n_reps))
    res_clusters = np.zeros((len(methods), nvv, n_reps))
    res_costs = np.zeros((len(methods), nvv, n_reps))
    res_lb = np.zeros((len(methods), nvv, n_reps))  # Lower bounds
    res_ub = np.zeros((len(methods), nvv, n_reps))  # Upper bound

    seeds = []
    for r in range(n_reps):
        seeds.append(np.random.randint(0, 1000000))

    t_beg = time.time()
    date_beg = dt.datetime.now()
    for i, val in enumerate(varied_values):
        p[varied_param] = val
        if varied_param == 'f':
            p['gap_cost'] = val
        print "\n{} {}.\n".format(val, varied_param)
        if cv:
            optimal_params = cross_validate_params(
                'LD', p['n_input_graphs'], p['n_entities'],
                p['n_input_graph_nodes'], p['p_keep_edge'],
                p['density_multiplier'], p['duplicates'], p['max_iters'], 1)
            p['f'] = optimal_params['f']
            optimal_params = cross_validate_params(
                'mKlau', p['n_input_graphs'], p['n_entities'],
                p['n_input_graph_nodes'], p['p_keep_edge'],
                p['density_multiplier'], p['duplicates'], p['max_iters'], 1)
            p['gap_cost'] = optimal_params['gap_cost']
        for r in range(n_reps):
            print "\n  Repetition: {}".format(r)
            seed = seeds[r]
            for j, method in enumerate(methods):
                print "\n  Method: {}\n".format(method)
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
                pr, rec, f1, o = single_cer(
                    p['f'], p['g'], p['gap_cost'], seed, method,
                    p['n_input_graphs'], p['n_entities'],
                    p['n_input_graph_nodes'], p['p_keep_edge'],
                    p['density_multiplier'], p['duplicates'], p['max_iters'],
                    mai, shuffle, max_entities)
                res_t[j, i, r] = time.time() - t0
                res_precision[j, i, r] = pr
                res_recall[j, i, r] = rec
                res_fscore[j, i, r] = f1
                res_iterations[j, i, r] = o['iterations']
                res_clusters[j, i, r] = o['n_clusters']
                res_costs[j, i, r] = o['cost']
                res_lb[j, i, r] = o['lb']
                res_ub[j, i, r] = o['ub']
    print "\nThe whole experiment took {:2f} seconds.".format(time.time() -
                                                              t_beg)

    fname = util.save_data(locals(), "synthetic_" + title)
    plot_toy_experiment_results(fname)

    print "F1 score:", np.mean(res_fscore, axis=2)
    print "Precision:", np.mean(res_precision, axis=2)
    print "Recall:", np.mean(res_recall, axis=2)
    print "Time:", np.mean(res_t, axis=2)
    print "Iterations:", np.mean(res_iterations, axis=2)
    print "Clusters:", np.mean(res_clusters, axis=2)
    print "Costs:", np.mean(res_costs, axis=2)
    print "Lower bounds:", np.mean(res_lb, axis=2)
    print "Upper bounds:", np.mean(res_ub, axis=2)


def test_single(f=None, do_plot=False, do_save=False, title=""):
    if f is None:
        if len(sys.argv) > 1:
            f = float(sys.argv[1])
        else:
            f = 0.4
    if len(sys.argv) > 2:
        max_entities = int(sys.argv[2])
    else:
        max_entities = None
    g = 0.5
    gap_cost = f
    mai = 1
    method = 'binB-LD'
    method = 'LD'
    #method = 'mKlau'
    #method = 'upProgmKlau'
    #method = 'progmKlau'
    seed = np.random.randint(0, 1000000)
    pr, re, f1, o1 = single_cer(
        f, g=g, gap_cost=gap_cost, seed=seed, method=method,
        n_input_graphs=10, n_duplicates=3, p_keep_edge=0.8,
        density_multiplier=1.1, n_entities=100, n_input_graph_nodes=30,
        max_iters=300, max_algorithm_iterations=mai, shuffle=True,
        max_entities=max_entities)

    if do_save:
        util.save_data(locals(), "single_synthetic_" + title)

    if do_plot:
        plt.plot(o1['Zd_scores'], '-x')
        plt.plot(o1['feasible_scores'], '-o')
        plt.show()

if __name__ == "__main__":
    test_single(f=0.2, do_plot=True, do_save=True, title="f0.2")
    test_single(f=1.2, do_plot=True, do_save=True, title="f1.2")
    f_effect(10)
