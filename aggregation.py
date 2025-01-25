#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numba import jit
import os
import time
#from BoxCounting import getFilledBoxes, boxCounting

@jit
def wrap_bounds(x, y, L):
    if (x >= L):
        x -= L
    if (x < 0):
        x += L
    if (y >= L):
        y -= L
    if (y < 0):
        y += L
    return x, y
# end

@jit(forceobj=True)
def coin_flip():
    return np.random.choice(["heads", "tails"])
# end

@jit(forceobj=True)
def walk(x, y, L):
    
    strCoinFlip = coin_flip()
    
    if strCoinFlip == "heads":
        x += np.random.choice([-1, 1])
    elif strCoinFlip == "tails":
        y += np.random.choice([-1, 1])
    
    x, y = wrap_bounds(x, y, L)
    
    return x, y
# end

@jit
def release_walker(L):
    # RELEASE THE HOUNDS
    side = np.random.randint(0, 4) # edge
    
    if side == 0:
        return 0, np.random.randint(0, L)
    elif side == 1:
        return L-1, np.random.randint(0, L)
    elif side == 2:
        return np.random.randint(0, L), L-1
    elif side == 3:
        return np.random.randint(0, L), 0
# end

@jit
def get_neighbors(x, y, L, S):
    neighbors = np.array([
            S[x, L - 1 if y == 0 else y - 1],
            S[x, 0 if y == L - 1 else y + 1],
            S[0 if x == L - 1 else x + 1, y],
            S[L - 1 if x == 0 else x - 1, y]
            ])
    return neighbors
# end

@jit
def stick(S, x, y):
    S[x, y] = 1
    return S
# end

@jit(forceobj=True)
def clear_folder(inFolder):
    files = os.listdir(inFolder)
    for fileName in files:
        fullPath = os.path.join(inFolder, fileName)
        os.remove(fullPath)
    # endfor
# end

@jit(forceobj=True)
def dla(L, clusterSitesMax, inSeed, sample, S):
    # init
    #N = L * L
    np.random.seed(inSeed)
    clusterSites = np.count_nonzero(S)
    moreSteps = True
    datafolder = "lattice"
    clear_folder(datafolder)
    
    x, y = release_walker(L)

    # main loop
    step = 0
    while (moreSteps):
        nns = get_neighbors(x, y, L, S)

        if np.count_nonzero(nns) == 0:
            x, y = walk(x, y, L)
        else:
            S = stick(S, x, y)
            clusterSites += 1
            x, y = release_walker(L)
            if clusterSites >= clusterSitesMax:
                moreSteps = False
                
            if clusterSites%sample == 0:
                filepath = os.path.join(datafolder, str(clusterSites))
                np.savetxt(filepath, S, delimiter=",")

        step += 1
    # end while    
    return S
# end

# init
L = 400
clusterSitesMax = 9000
seed = 876477
sample = 100

load_file = False

if load_file:
    S = np.loadtxt("4048", delimiter=",")
else:
    S = np.zeros((L,L))
    S[int(L/2), int(L/2)] = 1

# main loop
start = time.time()
S = dla(L, clusterSitesMax, seed, sample, S)
end = time.time()
print(end-start)

