#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 10:51:38 2020

Lalonde dataset

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

import warnings
warnings.filterwarnings("error")

from functions import *


### loading Lalonde's dataset rescaled as in the paper
X1_full = np.loadtxt('/Users/jeremylhour/Documents/code/pensynth/data/Lalonde_X1.txt',skiprows=1)
X0_full = np.loadtxt('/Users/jeremylhour/Documents/code/pensynth/data/Lalonde_X0.txt',skiprows=1)
Y0_full = np.loadtxt('/Users/jeremylhour/Documents/code/pensynth/data/Lalonde_Y0.txt',skiprows=1)

### Consolidate duplicates in X0
df = pd.DataFrame(X0_full)
df['outcome'] = Y0_full

# names
X_names = []
for j in range(1,11):
    X_names.append('var'+str(j))

df.columns = X_names + ['outcome']


X0_consolidated = df.groupby(X_names)[X_names + ['outcome']].mean()
X0 = X0_consolidated[X_names].to_numpy()
Y0 = X0_consolidated['outcome'].to_numpy()


### Applying the in-house algorithm
p = X1_full.shape[1]
all_w = np.zeros((X1_full.shape[0],p+1))

start_time = time.time()

for index in range(X1_full.shape[0]):
    sys.stdout.write("\r{0}".format(index))
    sys.stdout.flush()
    in_hull_flag = in_hull(X1_full[index], X0)
    try:
        if in_hull_flag:
            X0_tilde = incremental_pure_synth(X1_full[index],X0)
            w = pensynth_weights(np.transpose(X0_tilde),X1_full[index])
            all_w[index,] = np.transpose(w)

print(f"Temps d'exécution total : {(time.time() - start_time):.7f} secondes ---")