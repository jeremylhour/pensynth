#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 10:51:38 2020

@author: jeremylhour
"""

%reset

import sys
sys.path.append('/Users/jeremylhour/Documents/code/pensynth/incremental_algo_puresynth/')

import numpy as np
import pandas as pd
import time
import itertools
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from scipy.optimize import linprog

from functions import *


# Simulate data
n = 1001
p = 10

X = np.random.normal(0, 1, size=(n, p))
X1 = X[0]; X0 = np.delete(X, (0), axis=0)

in_hull_flag = in_hull(X1, X0)
print('Is inside the convex hull?: {}'.format(in_hull_flag))


# Method 1: Compute Delaunay Triangulation of X0
start_time = time.time()

tri = Delaunay(X0)
any_simplex = tri.find_simplex(X1)
print(any_simplex>=0)
the_simplex_Delaunay = tri.simplices[any_simplex]
print(X0[sorted(the_simplex_Delaunay),])

print(f"Temps d'exécution total : {(time.time() - start_time):.7f} secondes ---")


# Method 2: in-house algorithm
start_time = time.time()

simplex = incremental_pure_synth(X1,X0)
print(simplex)

print(f"Temps d'exécution total : {(time.time() - start_time):.7f} secondes ---")