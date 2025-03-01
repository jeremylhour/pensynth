"""
Functions used to run the in-house algorithm to
incrementally compute the Delaunay triangulation

Created on Sun Nov  1 15:03:32 2020
Refactored : 30/07/2021

@author: jeremylhour
"""

import numpy as np
import itertools
import time
from numba import njit

from scipy.spatial.distance import cdist
from scipy.optimize import linprog
from cvxopt import matrix, solvers
import warnings

from scipy.spatial import Delaunay

solvers.options["show_progress"] = False
solvers.options["abstol"] = 1e-8
solvers.options["reltol"] = 1e-8
solvers.options["maxiters"] = 500


# ------------------------------------------------------------------------------
# UNIT FUNCTIONS
# ------------------------------------------------------------------------------
@njit
def find_knn(node, nodes, k: int = 1):
    """
    find_knn:
        find the k nearest neighbors of node amongst nodes.

    Args:
        node (np.array): point for which we want to find the neighbors
        nodes (np.array): points that are candidate neighbors
        k (int): how many neighbors to return?

    Returns:
        np.array: the k nearest neighbors of node amongst nodes.
    """
    dist_2 = np.diag((nodes - node) @ (nodes - node).T)
    ranks = [sorted(dist_2).index(x) for x in dist_2]
    return nodes[[r <= k - 1 for r in ranks]]


@njit
def get_ranks(node, nodes):
    """
    get_ranks:
        returns the ranks and anti-ranks of nodes by rank in closeness to node.

    Args:
        node (np.array): point for which we want to find the neighbors
        nodes (np.array): points that are candidate neighbors

    Returns:
        np.array: the ranks of nodes by rank in closeness to node.
        np.array: the anti-ranks of nodes by rank in closeness to node.
    """
    dist_2 = np.diag((nodes - node) @ (nodes - node).T)
    ranks = np.array([sorted(dist_2).index(x) for x in dist_2])
    return ranks, np.argsort(ranks)


def is_in_hull(x, points):
    """
    is_in_hull:
        test if points in x are in hull

    Args:
        x (np.array): should be a (n, p) coordinates of n points in p dimensions
        points (np.array): the (m, p) array of the coordinates of m points in p dimensions

    Returns:
        bool: True if x is in the convex hull of points, False.
    """
    n_points = len(points)
    c = np.zeros(n_points)
    A = np.r_[points.T, np.ones((1, n_points))]
    b = np.r_[x, np.ones(1)]
    with warnings.catch_warnings():  # to ignore warning when degenerate cases
        warnings.simplefilter("ignore")
        lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success


def compute_radius_and_barycenter(nodes, fix_overflow: bool = True):
    """
    compute_radius_and_barycenter:
        returns radius, coordinates of barycenter
        for circumscribed hypersphere for these points

    Note: normally, it should return np.sqrt(-a[0]/2), a[1:] @ nodes, but overflow can occur so I force a positive value inside sqrt.

    Source:
        https://math.stackexchange.com/questions/1087011/calculating-the-radius-of-the-circumscribed-sphere-of-an-arbitrary-tetrahedron

    Args:
        nodes (np.array): array of dimension (p+1) x p of the p+1 points in p dimension
        fix_overflow (bool): if True brutally fixes the overflow

    Returns:
        float: radius of the circumscribed hypersphere
        np.array: coordinates of the barycenter
    """
    p = nodes.shape[1]

    Delta = np.zeros((p + 2, p + 2))
    Delta[0,] = np.concatenate(([0], np.ones(p + 1)), axis=0)
    Delta[:, 0] = np.concatenate(([0], np.ones(p + 1)), axis=0)
    Delta[1:, 1:] = cdist(nodes, nodes) ** 2

    a = np.linalg.inv(Delta)[:, 0]
    if fix_overflow:
        return np.sqrt(np.abs(a[0]) / 2), a[1:] @ nodes
    else:
        return np.sqrt(-a[0] / 2), a[1:] @ nodes


@njit
def is_inside_sphere(nodes, barycenter, radius: float):
    """
    inside_ball:
        find if any of the nodes is inside the given sphere

    Args:
        nodes (np.array): points to check if inside
        barycenter (np.array): coordinates of the barycenter
        radius (float): radius
    """
    dist_2 = np.diag((nodes - barycenter) @ (nodes - barycenter).T)
    return np.any(np.array([d < radius**2 for d in dist_2]))


@njit
def clip_to_zero(x, tol: float = 1e-5):
    """
    clip_to_zero:
        set values under threshold to zero.

    Args:
        x (np.array): numpy array of dimension 1.
        tol (float): tolerance.
    """
    x[x < tol] = 0.0
    return x / np.sum(x)


# ------------------------------------------------------------------------------
# MAIN FUNCTIONS
# ------------------------------------------------------------------------------
def incremental_pure_synth(X1, X0):
    """
    incremental_pure_synth:
        main algorithm, find the vertices of the simplex that X1 falls into
        returns the points and the antiranks

    Args:
        X1 (np.array): array of dimension p of the treated unit
        X0 (np.array): n x p array of untreated units

    """
    # get the ranks and anti-ranks of X0 with respect to their distances to X1
    _, antiRanks = get_ranks(X1, X0)
    n0, p = X0.shape

    # Initialize variables
    foundIt, inHullFlag = False, False

    # We loop over number of possible points, starting from p+1
    for k in range(p + 1, n0 + 1):
        # init the_simplex variable if there is a problem
        # when points is not inside convex hull, returns all the points
        the_simplex = tuple(range(k))

        # 1. Check if X1 belongs to the convex hull defined by its k nearest neighbors
        X_NN = X0[antiRanks[:k],]

        if not inHullFlag:
            inHullFlag = is_in_hull(X1, X_NN)

        if not inHullFlag:
            continue  # skip to next iteration if X1 not in convex hull of nearest neighbors

        # 2. For all the subsets of cardinality p+1 that have x in their convex hull...
        #  (since previous simplices did not contain X1,
        #   we need only to consider the simplices that have the new nearest neighbors as a vertex)
        # ...check if a point in X0 is contained in the circumscribing hypersphere of any of these simplices
        for item in itertools.combinations(range(k - 1), p):
            candidate = item + (k - 1,)
            if is_in_hull(X1, X_NN[candidate,]):
                try:
                    radius, center = compute_radius_and_barycenter(
                        X_NN[candidate,]
                    )  # sometimes gives an error if points have the same values for a particular X0
                except:
                    radius = np.nan

                if np.isnan(radius):  # if there is a degenerate case, we stop
                    the_simplex = candidate
                    foundIt = True
                    break

                if not is_inside_sphere(
                    np.delete(X0, antiRanks[candidate,], 0), center, radius
                ):
                    the_simplex = candidate
                    foundIt = True
                    break
        if foundIt:
            break

    antiRanks_tilde = sorted(antiRanks[the_simplex,])
    return X0[antiRanks_tilde,], antiRanks_tilde


def pensynth_weights(X0, X1, pen: float = 0.0, V=None):
    """
    pensynth_weights:
        computes penalized synthetic control weights with penalty pen

    See "A Penalized Synthetic Control Estimator for Disaggregated Data"

    Args:
        X0 (np.array): n x p matrix of untreated units
        X1 (np.array): 1 x p matrix of the treated unit
        pen (float): lambda, positive tuning parameter
        V (np.array): weights for the norm
    """
    if V is None:
        V = np.identity(X0.shape[1])
    n0 = len(X0)

    # OBJECTIVE
    delta = np.diag((X0 - X1) @ V @ (X0 - X1).T)
    P = matrix(X0 @ V @ X0.T)
    q = matrix(-X0 @ V @ X1 + pen * delta / 2)

    # ADDING-UP TO ONE
    A = matrix(1.0, (1, n0))
    b = matrix(1.0)

    # NON-NEGATIVITY
    G = matrix(-np.identity(n0))
    h = matrix(np.zeros(n0))

    # COMPUTE SOLUTION
    sol = solvers.qp(P, q, G, h, A, b)
    return clip_to_zero(np.squeeze(np.array(sol["x"])))


if __name__ == "__main__":
    # Test with simulated data
    n = 10
    p = 6

    X0 = np.random.normal(0, 1, size=(n, p))
    X1 = X0.mean(axis=0)

    is_in_hull_flag = is_in_hull(X1, X0)
    if is_in_hull_flag:
        print("Treated is inside convex hull.")
    else:
        print("Treated not in convex hull.")

    print("=" * 80)
    print("Method 1 : Compute Delaunay Triangulation of X0")
    print("=" * 80)

    start_time = time.time()
    tri = Delaunay(X0)
    any_simplex = tri.find_simplex(X1)
    print(any_simplex >= 0)
    delaunay_simplex = tri.simplices[any_simplex]
    print(X0[sorted(delaunay_simplex),])
    print(f"Runtime : {(time.time() - start_time):.2f} sec")

    print("=" * 80)
    print("Method 2 : incremental algorithm")
    print("=" * 80)

    start_time = time.time()
    simplex, _ = incremental_pure_synth(X1=X1, X0=X0)
    print(simplex)
    print(f"Runtime : {(time.time() - start_time):.2f} sec")
