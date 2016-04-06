import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle
import os
import networkx as nx
import matplotlib.cm as cmx
import matplotlib.colors as colors


title_map = {'res_fscore': 'F1 score',
             'res_objective': 'Cost',
             'res_t': 'Time (sec)',
             'res_iterations': 'Iterations until convergence',
             'res_precision': 'Precision',
             'res_recall': 'Recall',
             'res_clusters': '# of entities',
             'res_costs': 'Cost',
             'n_reps': 'Repetitions',
             'duality_gap': 'Duality gap',
             }

LD_lab = 'FLAN'
method_map = {'LD': LD_lab+'0', 'LD1': LD_lab+'0', 'LD5': LD_lab,
              'mKlau': 'Natalie', 'progmKlau': 'progNatalie',
              'upProgmKlau': 'progNatalie++', 'unary': 'Unary', 'meLD': 'cFLAN'}


def title_label(name):
    if name in title_map:
        return title_map[name]
    else:
        return name


def method_label(name):
    if name in method_map:
        return method_map[name]
    elif name.startswith('meLD'):
        parts = name.split('_')
        if len(parts) > 1:
            return 'cFLAN_' + parts[-1]
        else:
            return 'cFLAN'
    else:
        return name


def plot_toy_experiment_results(pickle_fname, show_plot=False,
                                pickle_fname2=None, for_print=False):
    """
    Plot synthetic data experiment results stored in a pickled file.
    """
    L = pickle.load(open(pickle_fname, 'rb'))
    print "Experiment seed:", L['experiment_seed']
    if pickle_fname2 is not None:
        L2 = pickle.load(open(pickle_fname2, 'rb'))
    else:
        L2 = None
    title_fields = ['n_input_graphs', 'duplicates', 'p_keep_edge',
                    'density_multiplier', 'f', 'g', 'n_entities',
                    'n_input_graph_nodes']
    if L['varied_param'] in title_fields:
        title_fields.remove(L['varied_param'])
    title_parts = ["{}={}".format(lab, L['params'][lab]) for lab in
                   title_fields] + ["{}={}".format('n_reps', L['n_reps'])]
    title = ", ".join(title_parts)# + '\n' + fname
    plotting_template(L, L['varied_param'], L['varied_values'], pickle_fname,
                      show_plot, title, L['params']['n_entities'], L2,
                      for_print=for_print)


def plot_multiplex_results(pickle_fname, show_plot=False, pickle_fname2=None,
                           for_print=False):
    L = pickle.load(open(pickle_fname, 'rb'))
    print "Experiment seed:", L['experiment_seed']
    vp = 'f'
    vv = L['f_values']
    title_parts = ["g={}, n_duplicate_names={}, n_reps={}".format(
        L['cost_params']['g'], L['duplicate_names'], L['n_reps'])]
    title = ", ".join(title_parts)# + '\n' + fname
    if pickle_fname2 is None:
        L2 = None
    else:
        L2 = pickle.load(open(pickle_fname2, 'rb'))
    plotting_template(L, vp, vv, pickle_fname, show_plot, title, L2=L2,
                      for_print=for_print)


def plot_genealogy_results(pickle_fname, show_plot=False, for_print=False):
    L = pickle.load(open(pickle_fname, 'rb'))
    vp = 'f'
    vv = L['f_vals']
    title_parts = ["n_trees={}, n_people={}, n_reps={}".format(
        L['n_trees'], L['n_people'], L['n_reps'])]
    title = ", ".join(title_parts)# + '\n' + fname
    plotting_template(L, vp, vv, pickle_fname, show_plot, title,
                      for_print=for_print)


def plotting_template(L, varied_param, varied_values, pickle_fname, show_plot,
                      title, n_entities=None, L2=None, for_print=False):

    plt.figure(figsize=(12, 6.5))
    plt.rc('font', family='serif')
    plt.rc('font', serif='Times New Roman')
    to_plot = ['res_precision', 'res_recall', 'res_fscore', 'res_clusters',
               'duality_gap', 'res_iterations']
    if not for_print:
        plt.suptitle(title)

    markers = ['-v', '-x', '-o', '-s', '-+', '-v', '-x', '-o', '-s', '-+']
    markers2 = ['--', '-', '--']
    #markers = ['-', 'b-', 'k-', 'm-', 'g-', 'c-', '-x', '-o', '-s', '-+']
    colors = 'rbkmgc'
    colors2 = 'rkb'
    # Maximum number of repetitions to plot (useful for plotting preliminary
    # results)
    max_n_reps = 1000

    for j, measure in enumerate(to_plot):
        if for_print and j == 5:
            continue
        plt.subplot(2, 3, j+1)
        ok_idxs = range(min(L['res_precision'].shape[2], max_n_reps))
        if measure == 'duality_gap':
            scores = np.mean(L['res_ub'][:, :, ok_idxs], axis=2) - \
                     np.mean(L['res_lb'][:, :, ok_idxs], axis=2)
        else:
            scores = np.mean(L[measure][:, :, ok_idxs], axis=2)
        #y_err = np.std(L[measure],axis=2)
        xs = varied_values
        all_ys = np.zeros((0, scores.shape[1]))
        for i, method in enumerate(L['methods']):
            if method in ('binB-LD3', 'ICM', 'clusterLD', 'unary++',
                          'upProgmKlau++'):
                continue
            if 'prog' in method.lower() and measure == 'duality_gap':
                continue
            #plt.errorbar(xs, scores[i,:], fmt='-o', label=method, yerr=y_err[i,:])
            #if L['varied_param'] == 'p_keep_edge':
            #    plt.semilogx(xs, scores[i,:], mfc='none', label=method_label(method))
            #else:
            ys = scores[i, :]
            m = markers[i]
            c = colors[i % len(colors)]
            if method.startswith('meLD'):
                # meLD outputs almost a flat line but due to random shufflings
                # there is small variation which we clean up here
                ys = [np.mean(scores[i, :])] * len(xs)
                m = '-'
                c = 'k'
            plt.plot(xs, ys, m, mfc='none', mec=c, color=c, alpha=0.6,
                     linewidth=1.5, label=method_label(method), markersize=5)
            all_ys = np.vstack((all_ys, ys))
        xd = max(xs) - min(xs)
        extra_space = 0.05
        plt.xlim([min(xs)-extra_space*xd, max(xs)+extra_space*xd])
        yd = np.max(all_ys) - np.min(all_ys)
        extra_space = 0.05
        plt.ylim([np.min(all_ys)-extra_space*yd, np.max(all_ys)+extra_space*yd])

        if L2 is not None:
            if measure == 'duality_gap':
                scores2 = np.mean(L2['res_ub'], axis=2) - \
                          np.mean(L2['res_lb'], axis=2)
            else:
                scores2 = np.mean(L2[measure], axis=2)
            for i, method in enumerate(L2['methods']):
                c = colors2[i % len(colors2)]
                m = markers2[i % len(markers2)]
                label = method_label(method)
                plt.plot(xs, [scores2[i, 0]] * len(xs), m, mfc='none', mec=c,
                         color=c, alpha=0.6, linewidth=1.5, label=label,
                         markersize=5)

        #if measure == 'res_clusters' and n_entities is not None:
        #    true_entities = n_entities
        #    plt.plot([min(xs), max(xs)], [true_entities, true_entities], 'k--')
        if measure in ['res_precision', 'res_recall', 'res_fscore']:
            min_y_tick = np.round(np.min(all_ys), decimals=1)
            max_y_tick = np.round(np.max(all_ys), decimals=1)
            #plt.ylim([0.45, np.max(all_ys)+extra_space*yd])
            #plt.yticks(np.arange(min_y_tick, max_y_tick+0.05, 0.1))

        plt.title(title_label(measure))
        #plt.xticks(np.arange(0,1.01,0.1))
        #plt.xlim([0, 1.3])
        if varied_param in ['f', 'g']:
            plt.xlabel("${}$".format(varied_param))
        else:
            plt.xlabel(varied_param)
        if j == 3 and for_print:
            plt.legend(bbox_to_anchor=(2.4, 1), loc=2, borderaxespad=0.)


    if not for_print:
        plt.subplot(2, 3, 6)
        plt.legend(loc=0)
    #plt.tight_layout()#pad=0.9, w_pad=0.1, h_pad=0.1)
    plt.subplots_adjust(hspace=.4)
    fig_fname = pickle_fname
    if fig_fname.endswith('.pckl'):
        fig_fname = fig_fname[:-5]
    fig_fname += '.pdf'
    plt.savefig(fig_fname)
    os.system('pdfcrop {} {}'.format(fig_fname, fig_fname))
    print "Saved:", fig_fname
    if show_plot:
        plt.show()


def plot_lb_ub(file_names):
    plt.figure(figsize=(8, 3.5))
    plt.rc('font', family='serif')
    plt.rc('font', serif='Times New Roman')
    for i, fn in enumerate(file_names):
        L = pickle.load(open(fn, 'rb'))
        lb = L['o1']['Zd_scores']
        ub = L['o1']['feasible_scores']

        plt.subplot(1, len(file_names), i+1)
        plt.plot(ub, color='r', alpha=0.6, linewidth=1.7,
                 label='Feasible solution')
        plt.plot(lb, color='b', alpha=0.6, linewidth=1.7,
                 label='Relaxed solution')
        plt.title("$f={}$".format(L['f']))
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.xlim([-2, 300])
        if i == 0:
            plt.legend(loc=4)
    plt.tight_layout()
    fig_fname = '../plots/ub_lb.pdf'
    plt.savefig(fig_fname)
    os.system('pdfcrop {} {}'.format(fig_fname, fig_fname))
    print "Saved:", fig_fname
    plt.show()


def plot_graphs(Gs):
    titles = ['$G_1$ (Lunch)', '$G_2$ (Facebook)', '$G_3$ (Co-author)',
              '$G_4$ (Leisure)', '$G_5$ (Work)', 'Entity graph']
    plt.figure(figsize=(10,6))
    plt.rc('font', family='serif')
    plt.rc('font', serif='Times New Roman')
    for gi, G in enumerate(Gs):
        plt.subplot(2, 3, gi+1)
        names = [G.node[v]['name']
                 for v in G]
        pos = nx.spring_layout(G, iterations=200, k=1.5)
        nx.draw(G, pos=pos, edge_color='silver', alpha=1,
                node_color=names, node_size=40, cmap=plt.cm.jet)
        plt.title(titles[gi])

    plt.tight_layout()
    fname = '../plots/multiplex_example.pdf'
    plt.savefig(fname)
    os.system('pdfcrop {} {}'.format(fname, fname))
    plt.show()


def main():
    fn = 'experiment_results/synthetic_f_effect_2016-03-26_134635.pckl'
    fn2 = 'experiment_results/synthetic_max_entities_2016-03-26_181059.pckl'
    plot_toy_experiment_results(fn, True, fn2, for_print=True)

    fn = 'experiment_results/multiplex_2016-03-28_050431.pckl'
    fn2 = 'experiment_results/multiplex_2016-03-28_124714.pckl'
    #fn = 'experiment_results/multiplex_2016-04-04_130958.pckl'
    #plot_multiplex_results(fn, True, fn2, for_print=True)

    fn = 'experiment_results/genealogical_2016-03-29_102118.pckl'
    fn = 'experiment_results/genealogical_2016-04-01_114534.pckl'
    #plot_genealogy_results(fn, True, for_print=False)

    fns = ['experiment_results/single_synthetic_f0.2_2016-03-28_000146.pckl',
           'experiment_results/single_synthetic_f1.2_2016-03-28_000350.pckl']
    #plot_lb_ub(fns)


if __name__ == '__main__':
    main()