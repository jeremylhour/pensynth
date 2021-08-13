#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Application to Lalonde (1986) dataset

Created on Sun Nov  1 10:51:38 2020

@author: jeremylhour
"""
import numpy as np
import pandas as pd
import time

from pensynthpy import in_hull, incremental_pure_synth, pensynth_weights


if __name__=='__main__':
    print('This is a script to compute the pure synthetic control solution for Lalonde (1986) data.')
    
    print("="*80)
    print("DATA MANAGEMENT")
    print("="*80)
    
    DATA_PATH = '../data/'
    
    ### Loading Lalonde's dataset rescaled as in the paper and unscaled for statistics
    X1_full = np.loadtxt(DATA_PATH+'X1.txt', skiprows=1)
    Y1_full = np.loadtxt(DATA_PATH+'Y1.txt', skiprows=1)
    X0_full = np.loadtxt(DATA_PATH+'X0.txt', skiprows=1)
    Y0_full = np.loadtxt(DATA_PATH+'Y0.txt', skiprows=1)
    X0_unscaled_full = np.loadtxt(DATA_PATH+'X0_unscaled.txt', skiprows=1)

    ### Consolidate duplicates in X0
    X_names = ['age', 'education', 'married', 'black', 'hispanic', 're74', 're75', 'nodegree', 'NoIncome74', 'NoIncome75']
    df = pd.DataFrame(X0_full)
    df.columns = [item+'_rescaled' for item in X_names]
    df['outcome'] = Y0_full
    
    for i in range(X0_unscaled_full.shape[1]):
        df[X_names[i]] = X0_unscaled_full[:,i]

    ### Consolidate dataset for untreated
    df_consolidated = df.groupby(X_names)[[i+'_rescaled' for i in X_names] + ['outcome'] + X_names].mean()
    X0 = df_consolidated[[i+'_rescaled' for i in X_names]].to_numpy()
    X0_unscaled = df_consolidated[X_names].to_numpy()
    Y0 = df_consolidated['outcome'].to_numpy()


    print("="*80)
    print("COMPUTING PURE SYNTHETIC CONTROL FOR EACH TREATED")
    print("="*80)
    
    # We proceed in 3 steps :
    # - if some untreated are the same as the treated, assign uniform weights to these untreated.
    # - if the treated is inside the convex hull defined by the untreated, run the incremental algo.
    # - if the treated is not inside the convex hull defined by the untreated, run the standard synthetic control.
    
    allW = np.zeros((len(X1_full), len(X0)))

    start_time = time.time()
    for index in range(len(X1_full)):
        print(f"Computing synthetic control for unit {index+1} out of {len(X1_full)}.")
        x = X1_full[index]
        sameAsUntreated = np.all(X0==x, axis=1) # True if untreated is same as treated

        if any(sameAsUntreated):
            untreatedId = [i for i, flag in enumerate(sameAsUntreated) if flag]
            allW[index, untreatedId] = 1/len(untreatedId)
        else:
            inHullFlag = in_hull(x=x, points=X0)
            if inHullFlag:
                X0_tilde, antiranks = incremental_pure_synth(X1=x, X0=X0)
                w = pensynth_weights(X0=np.transpose(X0_tilde), X1=x, pen=0)
                allW[index, antiranks] = np.transpose(w)
            else:
                w = pensynth_weights(X0=np.transpose(X0), X1=x, pen=1e-6)
                allW[index,] = np.transpose(w)
    print(f"Time elapsed : {(time.time() - start_time):.2f} seconds ---")


    print("="*80)
    print("SAVING RESULTS AND COMPUTING STATISTICS")
    print("="*80)
    
    np.savetxt('puresynth_solution.csv', allW, delimiter=',')

    ########## Compute the necessary statistics ##########
    Y0_hat = np.matmul(allW, Y0)

    print('ATT: {:.2f}'.format((Y1_full - Y0_hat).mean()))

    balance_check = np.matmul(allW, X0_unscaled).mean(axis=0)
    for b in range(len(balance_check)):
        print(X_names[b] +': {:.2f}'.format(balance_check[b]))

    sparsity_index = (allW > 0).sum(axis=1)
    print('Min sparsity: {:.0f}'.format(sparsity_index.min()))
    print('Median sparsity: {:.0f}'.format(np.median(sparsity_index)))
    print('Max sparsity: {:.0f}'.format(sparsity_index.max()))

    activ_index = (allW > 0).sum(axis=0)>0
    print('Active untreated units: {:.0f}'.format(activ_index.sum()))

    ########## Any unit with sparsity > p+1 ?  -- as a check ##########
    high_sparsity = np.where(sparsity_index>11)[0][0]
    print(f'{len(high_sparsity)} treated units have sparsity larger than p+1.')
    w_s = allW[high_sparsity,]