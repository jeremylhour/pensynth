#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 15:03:32 2020

Necessary functions

@author: jeremylhour
"""

import sys
import numpy as np
import itertools
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from scipy.optimize import linprog

def closest_points(node, nodes, k=1):
    """
    closest_points: find the k nearest neighbors
        
    :param node: point for which we want to find the neighbors
    :param nodes: point that are candidate neighbors
    :param k: how many neighbors to return?
    """
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    rank = [sorted(dist_2).index(x) for x in dist_2]
    return nodes[[r<=k-1 for r in rank]]

def get_ranks(node, nodes):
    """
    get_ranks: returns the ranks of nodes by rank in closeness to node
        
    :param node: point for which we want to find the neighbors
    :param nodes: point that are candidate neighbors
    """
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    rank = [sorted(dist_2).index(x) for x in dist_2]
    return rank


def in_hull(x, points):
    """
    in_hull: test if points in x are in hull

    :param x:  should be a n x p coordinates of n points in p dimensions
    :param points: he m x p array of the coordinates of m points in p dimensions 
    """
    n_points = len(points)
    n_dim = len(x)
    c = np.zeros(n_points)
    A = np.r_[points.T,np.ones((1,n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success


def compute_radius_and_barycenter(nodes):
    """
    compute_radius_and_barycenter:
        
    :param nodes: matrix of dimension p+1 x p
    
    See: https://math.stackexchange.com/questions/1087011/calculating-the-radius-of-the-circumscribed-sphere-of-an-arbitrary-tetrahedron
    """
    p = nodes.shape[1]
    theta = np.ones(p+1)
    Lambda = cdist(nodes,nodes)**2
    b = np.zeros(p+2); b[0] = 1
    
    Delta = np.zeros((p+2,p+2))
    Delta[0,] = np.concatenate(([0], theta), axis=0)
    Delta[:,0] = np.concatenate(([0], theta), axis=0)
    Delta[1:,1:] = Lambda
    
    a = np.linalg.inv(Delta)[:,0]
    return np.sqrt(-a[0]/2), np.matmul(a[1:],nodes)
      

def inside_sphere(nodes, barycenter, radius):
    """
    inside_ball: find any of the nodes is inside the given ball
        
    :param nodes: point to check if inside
    :param barycenter: coordinates of the barycenter
    :param radius: radius
    """
    deltas = nodes - barycenter
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return any([np.sqrt(d) < radius for d in dist_2])


def incremental_pure_synth(X1,X0):
    """
    incremental_pure_synth: find the vertices of the simplex that X1 falls into

    """
    # get the ranks of X0 with respect to their distances to X1
    ranks = np.array(get_ranks(X1,X0))
    anti_ranks = np.argsort(ranks)
    p = X0.shape[1]
    
    # Initialize variables
    found_it = False
    in_hull_flag = False
    
    ### BIG LOOP ###
    # find minimal k such that X1 is in convex hull
    for k in range(p+1, X0.shape[0]+1):   
        sys.stdout.write("\r{0}".format(k))
        sys.stdout.flush()
        # 1. Set of 'k' nearest neighbors
        X_NN = X0[anti_ranks[:k],]
        
        if not in_hull_flag:
            in_hull_flag = in_hull(X1, X_NN)
        
        if not in_hull_flag:
            continue # skip to next iteration if X1 not in convex hull of nearest neighbors
        
        # 2. Select all the subsets of cardinality p+1 that have x in their convex hull
        # Since previous simplices did not contain X1,
        #we need only to consider the simplices that ahev the new nearest neighbors as a vertex
        candidates = []
                
        for i in itertools.combinations(range(k-1),p):
            new_tuple = i + (k-1,)
            if in_hull(X1, X_NN[new_tuple,]):
                candidates.append(new_tuple)
                
        # 3. Check if a point in X0 is contained in the circumscribing hypersphere of any of these simplices
        for simplex in candidates:
            r, c = compute_radius_and_barycenter(X_NN[simplex,])
            if not inside_sphere(np.delete(X0,anti_ranks[simplex,],0), c, r):
                #print(simplex)
                the_simplex = simplex
                found_it = True
                break
        if found_it:
            break
    
    # Check
    return X0[sorted(anti_ranks[the_simplex,]),]


from cvxopt import matrix, solvers
from scipy.optimize import minimize


def pensynth_weights(X0, X1, pen = 0, **kwargs):
    V = kwargs.get('V', np.identity(X0.shape[0]))
    # OBJECTIVE
    n = X0.shape[1]
    delta = np.diag((X0 - np.reshape(np.repeat(X1,n,axis=0), (X1.shape[0],n))).T.dot(V.dot(X0 - np.reshape(np.repeat(X1,n,axis=0), (X1.shape[0],n)))))
    P = (X0.T).dot(V.dot(X0))
    P = matrix(P)
    q = -X0.T.dot(V.dot(X1)) + (pen/2)*delta
    q = matrix(q)
    # ADDING-UP
    A = matrix(1.0,(1,n))
    b = matrix(1.0)
    # NON-NEGATIVITY
    G = matrix(np.concatenate((np.identity(n),-np.identity(n))))
    h = matrix(np.concatenate((np.ones(n),np.zeros(n))))
    # SOLUTION
    sol = solvers.qp(P,q,G,h,A,b)
    return np.array(sol['x'])