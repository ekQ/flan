"""
NOTE: This method hasn't been used recently and therefor might not work as such.

Agglomerative clustering approach for multiple network alignment.
"""
import utilities as util


def agglomerative(asg, similarities, P, max_iters):
    """
    Agglomerative approach for multiple network alignment.

    TODO Get rid of similarities and use P instead
    """
    # Do some magic...
    changed = True
    iteration = 0
    while changed and iteration < max_iters:
        print "Iteration %d." % (iteration+1)
        changed = False
        clusters = asg.get_cluster_keys()
        min_cost = 1e20
        min_src_clust = None
        min_dst_clust = None
        for i, src_clust in enumerate(clusters):
            if len(similarities[src_clust]) <= 1:
                continue
            #print "Src clust:", src_clust, "|candidates|:", len(similarities[src_clust]), len(asg.get_candidates(src_clust))
            clusters_set = set(asg.get_cluster_keys())
            min_cost = 1e20
            min_src_clust = None
            min_dst_clust = None
            for dst_clust in asg.get_candidates(src_clust):
                if dst_clust not in clusters_set:
                    continue
                #test_asg = asg.copy()
                #test_asg.merge_clusters(src_clust, dst_clust)
                #c = util.cost(test_asg, P)

                updated_idxs = asg.update_all_matches(src_clust, dst_clust)
                c = util.cost(asg, P)
                asg.update_match(updated_idxs, src_clust)
                #print "Dst clust: %d, cost: %f" % (dst_clust, c)
                if c < min_cost:
                    min_cost = c
                    min_dst_clust = dst_clust
                    min_src_clust = src_clust
            if min_src_clust != min_dst_clust:
                #print "\tAssigning {} to {}.\tCost: {}".format(min_src_clust, min_dst_clust, min_cost)
                asg.merge_clusters(min_src_clust, min_dst_clust)
                changed = True
        iteration += 1
    print "Converged in %d iterations." % iteration
    return asg


def agglomerative_fixed(asg, similarities, P, max_entities):
    # Do some magic...
    changed = True
    iteration = 0
    n_open = asg.count_opened_entities()
    prev_cost = util.cost(asg, P)
    while n_open > max_entities:
        print "Iteration %d." % (iteration+1)
        clusters = asg.get_cluster_keys()
        min_cost = 1e20
        min_extra_cost = 1e20
        min_src_clust = None
        min_dst_clust = None
        min_n_merged = -1
        for i, src_clust in enumerate(clusters):
            if len(similarities[src_clust]) <= 1:
                continue
            #print "Src clust:", src_clust, "|candidates|:", len(similarities[src_clust]), len(asg.get_candidates(src_clust))
            clusters_set = set(asg.get_cluster_keys())
            for dst_clust in asg.get_candidates(src_clust):
                if dst_clust == src_clust or dst_clust not in clusters_set:
                    continue
                updated_idxs = asg.update_all_matches(src_clust, dst_clust)
                c = util.cost(asg, P)
                if c - prev_cost >= 0:
                    extra_cost = (c - prev_cost) / float(len(updated_idxs))
                else:
                    extra_cost = c - prev_cost
                #if len(updated_idxs) > 1:
                #    print "\t{}, {}".format(len(updated_idxs), extra_cost)
                asg.update_match(updated_idxs, src_clust)
                #print "Dst clust: %d, cost: %f" % (dst_clust, c)
                if extra_cost < min_extra_cost:
                    min_cost = c
                    min_extra_cost = extra_cost
                    min_dst_clust = dst_clust
                    min_src_clust = src_clust
                    min_n_merged = len(updated_idxs)
        asg.merge_clusters(min_src_clust, min_dst_clust)
        n_open = asg.count_opened_entities()
        print "\tAssigned {} ({}) to {}.\tCost: {}\tExtra cost: {}\tn_open: {}".format(min_src_clust, min_n_merged, min_dst_clust, min_cost, min_extra_cost, n_open)
        iteration += 1
        prev_cost = min_cost
    print "Converged in %d iterations." % iteration
    return asg
