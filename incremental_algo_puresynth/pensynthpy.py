#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions used to run the in-house algorithm to
incrementally compute the Delaunay triangulation

Created on Sun Nov  1 15:03:32 2020
Refactored : 30/07/2021

@author: jeremylhour
"""
import sys
import numpy as np
import math
import itertools
import time

from scipy.spatial.distance import cdist
from scipy.optimize import linprog
from cvxopt import matrix, solvers
import warnings

from scipy.spatial import Delaunay

# ------------------------------------------------------------------------------
# UTILS
# ------------------------------------------------------------------------------
def closest_points(node, nodes, k=1):
    """
    closest_points:
        find the k nearest neighbors
        
    @param node (np.array): point for which we want to find the neighbors
    @param nodes (np.array): points that are candidate neighbors
    @param k (int): how many neighbors to return?
    """
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    ranks = [sorted(dist_2).index(x) for x in dist_2]
    return nodes[[r<=k-1 for r in ranks]]

def get_ranks(node, nodes):
    """
    get_ranks:
        returns the ranks and anti-ranks of nodes by rank in closeness to node
        
    @param node (np.array): point for which we want to find the neighbors
    @param nodes (np.array): points that are candidate neighbors
    """
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    ranks = np.array([sorted(dist_2).index(x) for x in dist_2])
    return ranks, np.argsort(ranks)

def in_hull(x, points):
    """
    in_hull:
        test if points in x are in hull

    @param x (np.array):  should be a n x p coordinates of n points in p dimensions
    @param points (np.array): the m x p array of the coordinates of m points in p dimensions 
    """
    n_points = len(points)
    c = np.zeros(n_points)
    A = np.r_[points.T,np.ones((1,n_points))]
    b = np.r_[x, np.ones(1)]
    with warnings.catch_warnings(): # to ignore warning when degenerate cases
        warnings.simplefilter("ignore")
        lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success

def compute_radius_and_barycenter(nodes):
    """
    compute_radius_and_barycenter: 
        returns radius, coordinates of barycenter
        for hypersphere circumscribed hypersphere for these points
        
    @param nodes (np.array): array of dimension (p+1) x p of the p+1 points in p dimension
    
    Source:
        https://math.stackexchange.com/questions/1087011/calculating-the-radius-of-the-circumscribed-sphere-of-an-arbitrary-tetrahedron
    """
    p = nodes.shape[1]
    theta = np.ones(p+1)
    Lambda = cdist(nodes, nodes)**2
    b = np.concatenate(([1], np.zeros(p+1)))
    
    Delta = np.zeros((p+2,p+2))
    Delta[0,] = np.concatenate(([0], theta), axis=0)
    Delta[:,0] = np.concatenate(([0], theta), axis=0)
    Delta[1:,1:] = Lambda
    
    a = np.linalg.inv(Delta)[:,0]
    return np.sqrt(-a[0]/2), np.matmul(a[1:],nodes)
      
def inside_sphere(nodes, barycenter, radius):
    """
    inside_ball: find if any of the nodes is inside the given ball
        
    @param nodes (np.array): points to check if inside
    @param barycenter: coordinates of the barycenter
    @param radius (float): radius
    """
    deltas = nodes - barycenter
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return any([np.sqrt(d) < radius for d in dist_2])

def Tzero(w, tol=1e-5):
    """
    Tzero:
    set values under threshold to zero
    
    @param w (np.array): numpy array of dimension 1.
    @param tol (float): tolerance
    """
    w[w<tol] = 0
    return w/np.sum(w)


# ------------------------------------------------------------------------------
# MAIN FUNCTIONS
# ------------------------------------------------------------------------------
def incremental_pure_synth(X1,X0):
    """
    incremental_pure_synth: 
        main algorithm, find the vertices of the simplex that X1 falls into
        returns the points and the antiranks
    
    @param X1 (np.array): array of dimension p of the treated unit
    @param X0 (np.array): n x p array of untreated units

    """
    # get the ranks and anti-ranks of X0 with respect to their distances to X1
    ranks, antiRanks = get_ranks(X1, X0)
    N0, p = X0.shape
    
    # Initialize variables
    foundIt, inHullFlag = False, False
    
    ### BIG LOOP ###
    # We loop over number of possible points, starting from p+1
    for k in range(p+1, N0+1):
        sys.stdout.write("\r{0}".format(k))
        sys.stdout.flush()
        
        # init the_simplex variable if there is a problem
        # when points is not inside convex hull, returns all the points
        the_simplex =  tuple(range(k)) # init if there is a problem
        
        # 1. Set of 'k' nearest neighbors
        X_NN = X0[antiRanks[:k],]
        
        # 2. Check if X1 belongs to that convex hull
        if not inHullFlag:
            inHullFlag = in_hull(X1, X_NN)
        
        if not inHullFlag:
            continue # skip to next iteration if X1 not in convex hull of nearest neighbors
        
        # 3. For all the subsets of cardinality p+1 that have x in their convex hull...
        #  (since previous simplices did not contain X1,
        #   we need only to consider the simplices that have the new nearest neighbors as a vertex)
        # ...check if a point in X0 is contained in the circumscribing hypersphere of any of these simplices
        for i in itertools.combinations(range(k-1),p):
            candidate = i + (k-1,)
            if in_hull(X1, X_NN[candidate,]):
                r, c = compute_radius_and_barycenter(X_NN[candidate,]) # sometimes gives an error if points have the same values for a particular XÒ
                
                if math.isnan(r): # if there is a degenerate case, we stop
                    #the_simplex = tuple([i for i in range(k)]) # if there is a degenerate case, we stop
                    the_simplex = candidate #no? that will pre-select points...
                    foundIt = True
                    break
            
                if not inside_sphere(np.delete(X0, antiRanks[candidate,], 0), c, r):
                    the_simplex = candidate
                    foundIt = True
                    break
        if foundIt:
            break
            
    antiRanks_tilde = sorted(antiRanks[the_simplex,])
    return X0[antiRanks_tilde,], antiRanks_tilde


def pensynth_weights(X0, X1, pen=0, **kwargs):
    """
    pensynth_weights:
        computes penalized synthetic control weights with penalty pen
    
    @param X0 (np.array): p x n matrix of untreated units
    @param X1 (np.array): p x 1 matrix of the treated unit
    @param pen (float): lambda, positive parameter
    """
    V = kwargs.get('V', np.identity(len(X0)))
    # OBJECTIVE
    n = X0.shape[1]
    delta = np.diag((X0 - np.reshape(np.repeat(X1,n,axis=0), (len(X1),n))).T.dot(V.dot(X0 - np.reshape(np.repeat(X1,n,axis=0), (len(X1),n)))))
    P = (X0.T).dot(V.dot(X0))
    P = matrix(P)
    q = -X0.T.dot(V.dot(X1)) + (pen/2)*delta
    q = matrix(q)
    # ADDING-UP
    A = matrix(1.0,(1,n))
    b = matrix(1.0)
    # NON-NEGATIVITY
    G = matrix(np.concatenate((np.identity(n), -np.identity(n))))
    h = matrix(np.concatenate((np.ones(n), np.zeros(n))))
    # SOLUTION
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)
    return Tzero(np.array(sol['x']))


if __name__=='__main__':
    # Simulate data
    n = 21
    p = 5

    X = np.random.normal(0, 1, size=(n, p))
    X1 = X[0]
    X0 = np.delete(X, (0), axis=0)

    in_hull_flag = in_hull(X1, X0)
    if in_hull_flag:
        print("Treated is inside convex hull.")
    else:
        print("Treated not in convex hull.")
    
    print("="*80)
    print("Method 1: Compute Delaunay Triangulation of X0")
    print("="*80)
    
    start_time = time.time()
    tri = Delaunay(X0)
    any_simplex = tri.find_simplex(X1)
    print(any_simplex>=0)
    the_simplex_Delaunay = tri.simplices[any_simplex]
    print(X0[sorted(the_simplex_Delaunay),])
    print(f"Temps d'exécution total : {(time.time() - start_time):.7f} secondes ---")

    print("="*80)
    print("Method 2: in-house algorithm")
    print("="*80)
    
    start_time = time.time()
    simplex = incremental_pure_synth(X1,X0)
    print(simplex)
    print(f"Temps d'exécution total : {(time.time() - start_time):.7f} secondes ---")