"""
Generate toy graphs with a groundtruth entity graph.
"""
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def generate_graphs(N, n_subgraphs, n_subgraph_nodes, p_keep_edge=1,
                    density_multiplier=1, n_duplicate_names=5):
    assert n_subgraph_nodes <= N, "Subgraphs cannot be larger than the " + \
                                  "underlying graph"
    # Generate the underlying graph
    G = nx.barabasi_albert_graph(N,2)
    n_uniq_names = int(np.ceil(N / float(n_duplicate_names)))
    name_pool = []
    for i in range(n_uniq_names):
        name_pool += [i] * n_duplicate_names
    for node in G.nodes():
        G.node[node]['name'] = name_pool.pop(random.randint(0,len(name_pool)-1))
        #G.node[node]['entity'] = -1
    base_density = nx.density(G)
    #print "\nUnderlying graph density {}.\n".format(base_density)

    # Compute the edge adding probability so that density is correctly
    # multiplied
    assert density_multiplier >= 1, "Density multiplier must be at least 1."
    p_add_edge = base_density * (density_multiplier - 1)

    sGs = [] # Subgraphs
    for i in range(n_subgraphs):
        sG = nx.Graph()
        unexplored = [random.choice(G.nodes())]
        while len(sG) < n_subgraph_nodes and len(unexplored) > 0:
            next_node = random.choice(unexplored)
            sG.add_node(next_node, name=G.node[next_node]['name'],
                        entity=next_node, subgraph=i)
            neighs = G.neighbors(next_node)
            for neigh in neighs:
                if not sG.has_node(neigh):
                    if neigh not in unexplored:
                        unexplored.append(neigh)
                elif random.random() < p_keep_edge:
                    # Add neighbor
                    sG.add_edge(next_node, neigh)
            unexplored.remove(next_node)
        density0 = nx.density(sG)
        # Add edges to the subgraph
        if p_add_edge > 0:
            for u in sG.nodes():
                for v in sG.nodes():
                    if v <= u:
                        # Don't try adding the same edge twice
                        continue
                    if not sG.has_edge(u,v) and random.random() < p_add_edge:
                        sG.add_edge(u,v)
        density1 = nx.density(sG)
        #print "Subgraph {} density after removal {} and after adding {}."\
        #    .format(i, density0, density1)
        sGs.append(sG)
    return sGs, G


def draw_graphs(Gs):
    plt.figure(1)
    pos = nx.spring_layout(Gs[0])
    GG = nx.Graph()
    for i, G in enumerate(Gs):
        if i > 0:
            GG = nx.disjoint_union(GG, G)
        plt.subplot(1,len(Gs)+1,i+1)
        names = {j:"%d:%s" % (j,G.node[j]['name']) for j in G.nodes()}
        nx.draw(G, with_labels=True, labels=names, pos=pos)
    names = {j:"%d:%d" % (j,GG.node[j]['entity']) for j in GG.nodes()}
    plt.subplot(1,len(Gs)+1,len(Gs)+1)
    nx.draw(GG, with_labels=True, labels=names)
    plt.show()

if __name__ == '__main__':
    sGs_, G_ = generate_graphs(20, 2, 10, 1, 3)
    draw_graphs([G_]+sGs_)
