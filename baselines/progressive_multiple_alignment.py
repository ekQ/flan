"""
Align multiple networks progressively by decomposing the problem into multiple
pairwise alignment problems. Any pairwise method can be used for the
subproblems.
"""
import align_multiple_networks as amn
import networkx as nx


def progressive_multiple_alignment(
        Gs, cost_params, similarity_tuples, method, max_iters,
        max_algorithm_iterations, max_entities, shuffle, self_discount,
        do_evaluate, update_edges=False):
    """
    Merge input graphs one graph at a time using any 2-network alignment method.

    Input:
        (same as merge_multiple_networks())
    """
    assert len(Gs) > 1, "Must have at least 2 graphs to align."
    # Construct candidate matches data structure
    candidate_matches = {}
    if similarity_tuples is not None:
        for (g1, n1, g2, n2, sim) in similarity_tuples:
            if (g1, n1) not in candidate_matches:
                candidate_matches[(g1, n1)] = []
            candidate_matches[(g1, n1)].append((g2, n2, sim))
    else:
        # Collect candidate matches based on 'name' attributes of nodes
        for g1i, g1 in enumerate(Gs):
            for n1_label in g1.nodes():
                gn1 = (g1i, n1_label)
                # Every node can be mapped to itself
                candidate_matches[gn1] = [(g1i, n1_label, 1)]
                for g2i, g2 in enumerate(Gs):
                    if g2i == g1i:
                        continue
                    for n2_label in g2.nodes():
                        if g1.node[n1_label]['name'] == \
                                g2.node[n2_label]['name']:
                            candidate_matches[gn1].append((g2i, n2_label, 1))

    # Dict from (graph,node) -> master_graph_node
    node2idx = {}
    idx2node = {}
    # Set of nodes mapped to themselves (to which we might map future nodes)
    active_nodes = set()
    # The assignment / asg.matches
    alignment = {}
    # Add the first graph
    first_graph_idx = 0
    master_G = Gs[first_graph_idx]
    for node_idx, node_label in enumerate(master_G.nodes()):
        node2idx[(first_graph_idx, node_label)] = node_idx
        idx2node[node_idx] = (first_graph_idx, node_label)
        active_nodes.add((first_graph_idx, node_label))
        alignment[(first_graph_idx, node_label)] = (first_graph_idx, node_label)
    master_G = nx.relabel_nodes(master_G, {label: i for i, label in
                                           enumerate(master_G.nodes())})

    # Start aligning other graphs one at a time
    for graph_idx in range(len(Gs)):
        if graph_idx == first_graph_idx:
            continue
        print "Progressively aligning graph {}.".format(graph_idx)
        current_G = Gs[graph_idx]
        current_Gs = [master_G, current_G]
        # Construct a problem with only a subset of similarity_tuples
        temp_sim_tuples = []
        # Filter the similarity tuples between active_nodes and current graph
        for node_label in current_G:
            # Node can always be mapped to itself
            temp_sim_tuples.append((1, node_label, 1, node_label, 1))
            for cand_graph, cand_node, sim in candidate_matches[(graph_idx,
                                                                 node_label)]:
                if (cand_graph, cand_node) in active_nodes:
                    master_node_idx = node2idx[(cand_graph, cand_node)]
                    temp_sim_tuples.append((1, node_label, 0, master_node_idx,
                                            sim))
        x, other = amn.align_multiple_networks(
            current_Gs, cost_params, temp_sim_tuples, method, max_iters,
            max_algorithm_iterations, max_entities, shuffle, self_discount,
            do_evaluate)
        new_nodes = other['graph2nodes'][1]
        old_nodes = other['graph2nodes'][0]
        master_G = other['G']
        for node_i, node_idx in enumerate(new_nodes):
            node_label = current_G.nodes()[node_i]
            node2idx[(graph_idx, node_label)] = node_idx
            idx2node[node_idx] = (graph_idx, node_label)
            if x[node_idx] != node_idx:     # Map to master graph
                assert x[node_idx] in old_nodes, \
                    "Node should map to itself or to master graph"
            else:   # Map to itself
                active_nodes.add((graph_idx, node_label))
            alignment[(graph_idx, node_label)] = idx2node[x[node_idx]]
        if update_edges:
            # Update edge weights on master graph
            current_map = {label: i for i, label in enumerate(current_G.nodes())
                           }
            for u_label, v_label in current_G.edges():
                u = current_map[u_label]
                v = current_map[v_label]
                u_idx = new_nodes[u]
                v_idx = new_nodes[v]
                if x[u_idx] != u_idx or x[v_idx] != v_idx:
                    u_x = x[u_idx]
                    v_x = x[v_idx]

                    if v_x not in master_G.edge[u_x]:
                        master_G.add_edge(u_x, v_x, weight=1)
                    """
                    elif 'weight' in master_G.edge[u_x][v_x]:
                        # Weighted edges similar to FLAN (hasn't been properly
                        # tested yet)
                        master_G.edge[u_x][v_x]['weight'] += 1
                    """

    full_x = []
    for idx in master_G:
        graph_idx, node_idx = idx2node[idx]
        full_x.append(node2idx[alignment[(graph_idx, node_idx)]])
    results = {'best_x': full_x, 'feasible_scores': [-1], 'iterations': -1,
               'lb': -1, 'ub': -1}
    return results, master_G
