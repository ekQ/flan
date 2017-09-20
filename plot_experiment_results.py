import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle
import os
import networkx as nx
import matplotlib.cm as cmx
import matplotlib.colors as colors
import re


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
method_map = {'LD': LD_lab+'0', 'LD1': LD_lab+'0', 'LD5': LD_lab, 'binB-LD5': LD_lab,
              'mKlau': 'Natalie', 'progmKlau': 'progNatalie        ',
              'upProgmKlau': 'progNatalie++', 'unary': 'Unary', 'meLD': 'cFLAN',
              'isorankn': 'IsoRankN'}

MARKERSIZE = 4
LINEWIDTH = 1.1


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


def combine_pickle_files(data_dir):
    files = [f for f in os.listdir(data_dir) if f.endswith('.pckl')]
    print "{} files found.".format(len(files))
    matrices_to_concat = [
        'res_precision', 'res_recall', 'res_fscore', 'res_iterations', 'res_t',
        'res_clusters', 'res_costs', 'res_ub', 'res_lb']
    L = pickle.load(open(os.path.join(data_dir, files[0]), 'rb'))
    for f2 in files[1:]:
        L2 = pickle.load(open(os.path.join(data_dir, f2), 'rb'))
        L['n_reps'] += L2['n_reps']
        for mat in matrices_to_concat:
            if mat in L:
                L[mat] = np.concatenate((L[mat], L2[mat]), axis=2)
    L.pop('r', None)
    L.pop('j', None)
    return L


def plot_toy_experiment_results(pickle_fname, show_plot=False,
                                pickle_fname2=None, pickle_fname3=None,
                                for_print=False, yticks={}):
    """
    Plot synthetic data experiment results stored in a pickled file.
    :rtype: object
    """
    if os.path.isdir(pickle_fname):
        L = combine_pickle_files(pickle_fname)
    else:
        L = pickle.load(open(pickle_fname, 'rb'))
    print "Experiment seed:", L['experiment_seed']
    if pickle_fname2 is not None:
        L2 = pickle.load(open(pickle_fname2, 'rb'))
    else:
        L2 = None
    if pickle_fname3 is not None:
        L3 = pickle.load(open(pickle_fname3, 'rb'))
    else:
        L3 = None
    title_fields = ['n_input_graphs', 'duplicates', 'p_keep_edge',
                    'density_multiplier', 'f', 'g', 'n_entities',
                    'n_input_graph_nodes']
    if L['varied_param'] in title_fields:
        title_fields.remove(L['varied_param'])
    title_parts = ["{}={}".format(lab, L['params'][lab]) for lab in
                   title_fields] + ["{}={}".format('n_reps', L['n_reps'])]
    title = ", ".join(title_parts)# + '\n' + fname
    plotting_template(L, L['varied_param'], L['varied_values'], pickle_fname,
                      show_plot, title, L['params']['n_entities'], L2, L3,
                      for_print=for_print, yticks=yticks)


def plot_multiplex_results(pickle_fname, show_plot=False, pickle_fname2=None,
                           pickle_fname3=None, for_print=False, ylims={},
                           yticks={}):
    if os.path.isdir(pickle_fname):
        L = combine_pickle_files(pickle_fname)
    else:
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
    if pickle_fname3 is None:
        L3 = None
    else:
        L3 = pickle.load(open(pickle_fname3, 'rb'))
    plotting_template(L, vp, vv, pickle_fname, show_plot, title, L2=L2, L3=L3,
                      for_print=for_print, ylims=ylims, yticks=yticks)


def plot_genealogy_results(pickle_fname, show_plot=False, pickle_fname3=None,
                           for_print=False):
    if os.path.isdir(pickle_fname):
        L = combine_pickle_files(pickle_fname)
    else:
        L = pickle.load(open(pickle_fname, 'rb'))
    vp = 'f'
    vv = L['f_vals']
    title_parts = ["n_trees={}, n_people={}, n_reps={}".format(
        L['n_trees'], L['n_people'], L['n_reps'])]
    title = ", ".join(title_parts)# + '\n' + fname
    if pickle_fname3 is None:
        L3 = None
    else:
        L3 = pickle.load(open(pickle_fname3, 'rb'))
    plotting_template(L, vp, vv, pickle_fname, show_plot, title, L3=L3,
                      for_print=for_print)


def texify(text):
    text = re.sub('#', '\#', text)
    text = re.sub('_', '\_', text)
    return text


def plotting_template(L, varied_param, varied_values, pickle_fname, show_plot,
                      title, n_entities=None, L2=None, L3=None,
                      for_print=False, ylims={}, yticks={}):

    #plt.figure(figsize=(12, 6.5))
    plt.figure(figsize=(8, 9))
    plt.rc('font', family='serif')
    plt.rc('font', serif='Times New Roman')
    plt.rc('text', usetex=True)
    fs = 12 # Font size
    if not for_print:
        plt.suptitle(title)
    to_plot = ['res_precision', 'res_recall', 'res_fscore', 'res_clusters',
               'duality_gap', 'res_iterations']
    markers = ['-v', '-s', '-s', '-x', '-+', '-v', '-x', '-o', '-s', '-+', '-v']
    #markers = ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-']
    markers2 = ['--', '-', '--']
    markers3 = ['-.', '-', '--']
    #markers = ['-', 'b-', 'k-', 'm-', 'g-', 'c-', '-x', '-o', '-s', '-+']
    colors = 'rbcmgk'
    colors2 = 'rkb'
    colors3 = 'm'
    # Maximum number of repetitions to plot (useful for plotting preliminary
    # results)
    max_n_reps = L['n_reps']
    print "methods:", L['methods']

    for j, measure in enumerate(to_plot):
        if for_print and j == 5:
            continue
        #plt.subplot(2, 3, j+1)
        ax = plt.subplot(3, 2, j+1)
        if 'r' in L:
            completed_reps = L['r'] + 1
        elif 'j' in L:
            completed_reps = L['j'] + 1
        else:
            completed_reps = max_n_reps
        if completed_reps < max_n_reps:
            print "Plotting only {} repetitions.".format(completed_reps)
        ok_idxs = range(min(L['res_precision'].shape[2], completed_reps))
        if measure == 'duality_gap':
            scores = np.mean(L['res_ub'][:, :, ok_idxs], axis=2) - \
                     np.mean(L['res_lb'][:, :, ok_idxs], axis=2)
        else:
            scores = np.mean(L[measure][:, :, ok_idxs], axis=2)
        #y_err = np.std(L[measure],axis=2)
        all_ys = []
        for i, method in enumerate(L['methods']):
            if method in ('binB-LD3', 'ICM', 'clusterLD', 'unary++',
                          'upProgmKlau++', 'LD5'):
                continue
            if ('prog' in method.lower() or method == 'isorankn') and \
                    measure == 'duality_gap':
                continue
            #plt.errorbar(xs, scores[i,:], fmt='-o', label=method, yerr=y_err[i,:])
            #if L['varied_param'] == 'p_keep_edge':
            #    plt.semilogx(xs, scores[i,:], mfc='none', label=method_label(method))
            #else:
            xs = varied_values
            ys = scores[i, :]
            m = markers[i]
            c = colors[i % len(colors)]
            if method.startswith('meLD'):
                # meLD outputs almost a flat line but due to random shufflings
                # there is small variation which we clean up here
                ys = [scores[i, 0]] * len(xs)
                if method in ('meLD5_61', 'meLD5_100'):
                    m = '-'
                    c = 'k'
                elif method in ('meLD5_50', 'meLD5_75'):
                    m = '--'
                    c = 'r'
                elif method in ('meLD5_70', 'meLD5_125'):
                    m = '--'
                    c = 'b'
                else:
                    m = '-'
                    c = 'k'
            elif method == 'isorankn':
                xs = np.asarray(xs)
                ys = np.asarray(ys)
                #print "isorankn min max:", min(ys), max(ys)
                ys = ys[xs<=1]
                xs = xs[xs<=1]
                m = '-o'
                c = 'm'
            ax.plot(xs, ys, m, mfc='none', mec=c, color=c, alpha=0.6,
                    linewidth=LINEWIDTH, label=texify(method_label(method)),
                    markersize=MARKERSIZE, markeredgewidth=LINEWIDTH)
            #print all_ys.shape, ys.shape
            all_ys = np.hstack((all_ys, ys))
        min_y = np.min(all_ys)
        max_y = np.max(all_ys)

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
                y = scores2[i, 0]
                ax.plot(xs, [y] * len(xs), m, mfc='none', mec=c,
                        color=c, alpha=0.6, linewidth=LINEWIDTH, label=texify(label),
                        markersize=MARKERSIZE, markeredgewidth=LINEWIDTH)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
        if L3 is not None:
            if measure == 'duality_gap':
                scores3 = np.mean(L3['res_ub'], axis=2) - \
                          np.mean(L3['res_lb'], axis=2)
            else:
                scores3 = np.mean(L3[measure], axis=2)
            for i, method in enumerate(L3['methods']):
                c = colors3[i % len(colors3)]
                m = markers3[i % len(markers3)]
                label = method_label(method)
                idx = 1     # Update this index for a different alpha value
                #print L3.keys()
                #print "scores3 size:", scores3.shape, scores3, L3['varied_param'], L3['f_values'] if 'f_values' in L3 else L3['varied_values']
                y = scores3[i, idx]
                ax.plot(xs, [y] * len(xs), m, mfc='none', mec=c,
                        color=c, alpha=0.6, linewidth=LINEWIDTH, label=label,
                        markersize=MARKERSIZE, markeredgewidth=LINEWIDTH)
                min_y = min(min_y, y)
                max_y = max(max_y, y)

        xs = varied_values
        xd = max(xs) - min(xs)
        extra_space = 0.05
        ax.set_xlim([min(xs)-extra_space*xd, max(xs)+extra_space*xd])
        if j not in ylims:
            yd = max_y - min_y
            extra_space = 0.05
            ax.set_ylim([min_y-extra_space*yd, max_y+extra_space*yd])
        else:
            ax.set_ylim(ylims[j])

        #if measure == 'res_clusters' and n_entities is not None:
        #    true_entities = n_entities
        #    plt.plot([min(xs), max(xs)], [true_entities, true_entities], 'k--')
        if measure in ['res_precision', 'res_recall', 'res_fscore']:
            min_y_tick = np.round(np.min(all_ys), decimals=1)
            max_y_tick = np.round(np.max(all_ys), decimals=1)
            #plt.ylim([0.40, np.max(all_ys)+extra_space*yd])
            #plt.yticks(np.arange(min_y_tick, max_y_tick+0.05, 0.1))

        if j in yticks:
            ax.set_yticks(yticks[j])

        ax.set_ylabel(texify(title_label(measure)), fontsize=fs)
        #plt.xticks(np.arange(0,1.01,0.1))
        #plt.xlim([0, 1.3])
        if varied_param in ['f', 'g']:
            ax.set_xlabel("${}$".format(varied_param), fontsize=fs)
        else:
            ax.set_xlabel(texify(varied_param), fontsize=fs)
        if j == 3 and for_print:
            #plt.legend(bbox_to_anchor=(1.4, 1), loc=2, borderaxespad=0.)
            ax.legend(bbox_to_anchor=(0, -0.4), loc=2, borderaxespad=0.,
                      prop={'size': fs})
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


    if not for_print:
        #plt.subplot(2, 3, 6)
        ax = plt.subplot(3, 2, 6)
        ax.legend(loc=0)
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
    plt.rc('text', usetex=True)
    for i, fn in enumerate(file_names):
        L = pickle.load(open(fn, 'rb'))
        lb = L['o1']['Zd_scores']
        ub = L['o1']['feasible_scores']

        ax = plt.subplot(1, len(file_names), i+1)
        ax.plot(ub, color='r', alpha=0.6, linewidth=1.7,
                label='Feasible solution')
        ax.plot(lb, color='b', alpha=0.6, linewidth=1.7,
                label='Relaxed solution')
        ax.set_title("$f={}$".format(L['f']), fontsize=13)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Cost', fontsize=12)
        ax.set_xlim([-2, 300])
        if i == 0:
            ax.legend(loc=4, fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig_fname = '../plots/ub_lb.pdf'
    plt.savefig(fig_fname)
    os.system('pdfcrop {} {}'.format(fig_fname, fig_fname))
    print "Saved:", fig_fname
    plt.show()


def plot_scalability_results(pickle_fname, pickle_fname2, show_plot=False,
                             for_print=False):
    L = pickle.load(open(pickle_fname, 'rb'))
    title_parts = ["n_people={}, f={}, n_reps={}".format(L['n_people'], L['f'],
                                                         L['n_reps'])]
    title = ", ".join(title_parts)# + '\n' + fname

    plt.figure(figsize=(9, 3))
    plt.rc('font', family='serif')
    plt.rc('font', serif='Times New Roman')
    plt.rc('text', usetex=True)
    fs = 12 # Font size.
    if not for_print:
        plt.suptitle(title)
    markers = ['-s', '-s', '-x', '-+', '-v', '-x', '-+', '-s', '-+', '-v']
    #markers = ['-x', '-o', '-s', '-+', '-v', '-x', '-o', '-s', '-+', '-v']
    #markers = ['-', 'b-', 'k-', 'm-', 'g-', 'c-', '-x', '-o', '-s', '-+']
    colors = 'bcmgkr'
    # Maximum number of repetitions to plot (useful for plotting preliminary
    # results)
    max_n_reps = L['n_reps']
    print "Maximum number of reps:", max_n_reps

    measure = 'res_t'
    ax = plt.subplot(1, 3, 1)
    varied_values = L['params']
    varied_param = 'Number of graphs\n(100 people per graph)'
    if 'r' in L:
        completed_reps = L['r'] + 1
    else:
        completed_reps = L['j'] + 1
    if completed_reps < max_n_reps:
        print "Plotting only {} repetitions.".format(completed_reps)
    ok_idxs = range(min(L['res_precision'].shape[2], completed_reps))
    scores = np.mean(L[measure][:, :, ok_idxs], axis=2)
    #y_err = np.std(L[measure],axis=2)
    xs = varied_values
    all_ys = np.zeros((0, scores.shape[1]))
    for i, method in enumerate(L['methods']):
        if method in ('binB-LD3', 'ICM', 'clusterLD', 'unary++',
                      'upProgmKlau++'):
            continue
        ys = scores[i, :]
        m = markers[i]
        c = colors[i % len(colors)]
        if method == 'isorankn':
            m = '-o'
            c = 'm'
        ax.plot(xs, ys, m, mfc='none', mec=c, color=c, alpha=0.6,
                linewidth=LINEWIDTH, label=method_label(method),
                markersize=MARKERSIZE, markeredgewidth=LINEWIDTH)
        all_ys = np.vstack((all_ys, ys))

    xd = max(xs) - min(xs)
    extra_space = 0.05
    ax.set_xlim([min(xs)-extra_space*xd, max(xs)+extra_space*xd])
    min_y = np.min(all_ys)
    max_y = np.max(all_ys)
    yd = max_y - min_y
    extra_space = 0.05
    ax.set_ylim([min_y-extra_space*yd, max_y+extra_space*yd])

    ax.set_ylabel(title_label(measure), fontsize=fs)
    ax.set_xlabel(varied_param, fontsize=fs)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Second plot
    L2 = pickle.load(open(pickle_fname2, 'rb'))
    max_n_reps = L2['n_reps']
    print "Maximum number of reps2:", max_n_reps

    measure = 'res_t'
    ax = plt.subplot(1, 3, 2)
    varied_values = L2['params']
    varied_param = 'Number of people per graph\n(4 graphs)'
    if 'r' in L2:
        completed_reps = L2['r'] + 1
    else:
        completed_reps = L2['j'] + 1
    if completed_reps < max_n_reps:
        print "Plotting only {} repetitions.".format(completed_reps)
    ok_idxs = range(min(L2['res_precision'].shape[2], completed_reps))
    scores = np.mean(L2[measure][:, :, ok_idxs], axis=2)
    xs = varied_values
    all_ys = np.zeros((0, scores.shape[1]))
    for i, method in enumerate(L2['methods']):
        if method in ('binB-LD3', 'ICM', 'clusterLD', 'unary++',
                      'upProgmKlau++'):
            continue
        if 'prog' in method.lower() and measure == 'duality_gap':
            continue
        ys = scores[i, :]
        if method == 'progmKlau':
            print "progNatalie:", xs[-1], ys[-1]
        m = markers[i]
        c = colors[i % len(colors)]
        if method == 'isorankn':
            m = '-o'
            c = 'm'
        ax.plot(xs, ys, m, mfc='none', mec=c, color=c, alpha=0.6,
                linewidth=LINEWIDTH, label=method_label(method),
                markersize=MARKERSIZE, markeredgewidth=LINEWIDTH)
        all_ys = np.vstack((all_ys, ys))
    min_y = np.min(all_ys)
    max_y = np.max(all_ys)

    xd = max(xs) - min(xs)
    extra_space = 0.05
    ax.set_xlim([min(xs)-extra_space*xd, max(xs)+extra_space*xd])
    yd = max_y - min_y
    extra_space = 0.05
    ax.set_ylim([min_y-extra_space*yd, max_y+extra_space*yd])

    ax.set_ylabel(title_label(measure), fontsize=fs)
    ax.set_xlabel(varied_param, fontsize=fs)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    if for_print:
        ax = plt.subplot(1, 3, 1)
        ax.legend(bbox_to_anchor=(2.5, 1), loc=2, borderaxespad=0., fontsize=fs)#, borderpad=1)
    plt.tight_layout()#pad=0.9, w_pad=0.1, h_pad=0.1)
    #plt.subplots_adjust(hspace=.9)
    fig_fname = pickle_fname
    if fig_fname.endswith('.pckl'):
        fig_fname = fig_fname[:-5]
    fig_fname += '.pdf'
    plt.savefig(fig_fname)
    os.system('pdfcrop {} {}'.format(fig_fname, fig_fname))
    print "Saved:", fig_fname
    if show_plot:
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
    # The result data is not available in the Git repository but can be
    # obtained by contacting the author.
    fn = 'experiment_results/toy0'
    fn2 = None
    fn3 = None
    plot_toy_experiment_results(fn, True, fn2, fn3, for_print=True,
                                yticks={0: np.arange(0.4, 0.66, 0.05),
                                        2: np.arange(0.1, 0.61, 0.1)})

    fn = 'experiment_results/multiplex1'
    fn2 = None
    fn3 = None
    plot_multiplex_results(fn, True, fn2, fn3, for_print=True,
                           ylims={0:(0.47,0.69), 1:(0.03,0.77), 2:(0.03,0.77)},
                           yticks={1:np.arange(0.1,0.71,0.1),
                                   2:np.arange(0.1,0.71,0.1)})

    fn = 'experiment_results/genealogy0'
    fn3 = None
    plot_genealogy_results(fn, True, pickle_fname3=fn3, for_print=True)

    fns = ['experiment_results/single_synthetic_f0.2_2016-03-28_000146.pckl',
           'experiment_results/single_synthetic_f1.2_2016-03-28_000350.pckl']
    plot_lb_ub(fns)

    fn = 'experiment_results/scalability_vary_trees_part_2016-09-12_215305.pckl'
    fn2 = 'experiment_results/scalability_vary_people_2016-10-11_034824.pckl'
    plot_scalability_results(fn, fn2, True, True)


if __name__ == '__main__':
    main()
