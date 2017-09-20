"""
Iterative conditional modes method for multiple network alignment.
"""
import random
import utilities as util


def ICM(asg, similarities, P, max_iters):
    # Do some magic...
    changed = True
    iteration = 0
    min_cost = 1e20
    while changed and iteration < max_iters:
        print "Iteration %d. Min cost: %.5f" % (iteration+1, min_cost)
        changed = False
        items = list(enumerate(asg.matches))
        random.shuffle(items)
        for i, src_clust in items:
            dst_candidates = asg.get_candidates(i)
            if len(dst_candidates) <= 1:
                #print "\tOnly %d candidates for %d!" % (len(dst_candidates), i)
                if len(dst_candidates) == 1 and src_clust != dst_candidates[0]:
                    asg.update_match(i, dst_clust)
                continue
            #print "Src clust:", src_clust, "|candidates|:", len(similarities[src_clust]), len(asg.get_candidates(src_clust))
            min_cost = 1e20
            min_dst_clust = None
            for dst_clust in dst_candidates:
                #test_asg = asg.copy()
                #test_asg.merge_clusters(src_clust, dst_clust)
                #c = util.cost(test_asg, P)

                #asg.update_match(i, dst_clust)
                #c = util.cost(asg, P)
                #asg.update_match(i, src_clust)
                #c = util.delta_cost(i, dst_clust, asg, P)
                c = util.delta_cost2(i, dst_clust, asg, P)
                #print "Dst clust: %d, cost: %f" % (dst_clust, c)
                if c < min_cost:
                    min_cost = c
                    min_dst_clust = dst_clust
            if src_clust != min_dst_clust:
                #print "\tAssigning {} to {}.\tCost: {}".format(min_src_clust, min_dst_clust, min_cost)
                asg.update_match(i, min_dst_clust)
                changed = True
        iteration += 1
    print "Converged in %d iterations." % iteration
    return asg, min_cost, iteration