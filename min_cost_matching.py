from scipy.optimize import linear_sum_assignment
#import dlib  # DLIB is faster than scipy.
import numpy as np


def min_cost_matching(cost_array, cost_infeasible=1e10, solver='scipy'):#dlib'):
    if solver == 'scipy':
        return linear_sum_assignment(cost_array)
    elif solver == 'dlib':
        n_rows, n_cols = cost_array.shape
        #print "(n_rows, n_cols) = ({}, {})".format(n_rows, n_cols)
        if n_rows < n_cols:
            cost_array = np.vstack(
                (cost_array, np.ones((n_cols-n_rows, n_cols))*cost_infeasible))
        elif n_rows > n_cols:
            cost_array = np.hstack(
                (cost_array, np.ones((n_rows, n_rows-n_cols))*cost_infeasible))
        cost = dlib.matrix((-cost_array).tolist())
        assignment = dlib.max_cost_assignment(cost)
        if n_rows < n_cols:
            assignment = assignment[:n_rows]
            idxs = range(n_rows)
        elif n_rows > n_cols:
            new_asg = []
            idxs = []
            for i, av in enumerate(assignment):
                if av < n_cols:
                    idxs.append(i)
                    new_asg.append(av)
            assignment = new_asg
        else:
            idxs = range(n_rows)

        return np.asarray(idxs), np.asarray(assignment)
