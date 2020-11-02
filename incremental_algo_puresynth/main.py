#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 10:51:38 2020

Lalonde dataset

@author: jeremylhour
"""


########## FUNCTIONS AND PACKAGES ##########

import sys
sys.path.append('/Users/jeremylhour/Documents/code/pensynth/incremental_algo_puresynth/')

import numpy as np
import pandas as pd
import time

from functions import *

########## DATA MANAGEMENT ##########

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


########## APPLYING THE IN-HOUSE ALGORITHM ##########

p = X1_full.shape[1]
all_w = np.zeros((X1_full.shape[0],X0_consolidated.shape[0]))

start_time = time.time()

for index in range(X1_full.shape[0]):
    sys.stdout.write("\r{0}".format(index))
    sys.stdout.flush()
    
    x = X1_full[index]
    same_as_untreated = np.all(X0==x,axis=1) # True if untreated is same as treated
    
    if any(same_as_untreated): # if same as treated, assign uniform weights to these untreated
        untreated_id = [i for i, x in enumerate(same_as_untreated) if x]
        all_w[index,untreated_id] = 1/len(untreated_id)
    else:
        in_hull_flag = in_hull(x, X0)
        if in_hull_flag:
            X0_tilde, antiranks = incremental_pure_synth(X1_full[index],X0)
            w = pensynth_weights(np.transpose(X0_tilde),X1_full[index])
            all_w[index,antiranks] = np.transpose(w)
        else:
            w = pensynth_weights(np.transpose(X0),X1_full[index])
            all_w[index,] = np.transpose(w)

print(f"Temps d'exécution total : {(time.time() - start_time):.2f} secondes ---")