#!/usr/bin/env python

# built-in libraries
from copy import deepcopy
import time
# my files
from bd_resample import *
from reg_bd_svd import *
from reg_bd3 import *
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import os


def bdreg(B, N=None, VamModel=None, BuildModel=None):
    print('## bdreg.py')
    np.set_printoptions(precision=5, suppress=True)
    if N is None:
        N = 50
    if not BuildModel:
        print('applying model')
        N = VamModel['N']
    elif BuildModel:
        print('building model')
        VamModel['N'] = N
    kll = len(B)
    bdpc = np.zeros([kll, 2 * N])
    bdpc0 = deepcopy(bdpc)
    sc = np.zeros([kll, 1])

    num_cores = multiprocessing.cpu_count()
    print('available cpu cores : ', num_cores)
    for ktt in range(kll):  # speed : 3 sec
        bdt = bd_resample((B.loc[ktt]), N)
        B.loc[ktt], sc[ktt] = reg_bd_svd(bdt)
        bdpc0[ktt] = np.append([B[ktt][1]], [B[ktt][0]], axis=1)

    mbdpc0 = [sum(x) / len(x) for x in zip(*bdpc0)]
    bdr0 = np.append([mbdpc0[N:]], [mbdpc0[0:N]], axis=0)

    if BuildModel:
        bdrn = deepcopy(bdr0)
        VamModel['bdrn'] = bdrn
    else:
        bdrn = VamModel['bdrn']

    # bnreg = deepcopy(B)
    # for ktt in range(kll):  # speed : 60 sec
    #     bnreg[ktt] = reg_bd3(bnreg.loc[ktt], bdrn)
    #     bdpc[ktt] = np.append(bnreg[ktt][1], bnreg[ktt][0])

    start = time.time()
    bnreg_a = deepcopy(B)
    bnreg2 = Parallel(n_jobs=num_cores)(delayed(reg_bd3)(bnreg_a.loc[kk], bdrn) for kk in range(kll))
    bdpc2 = [np.append(bnreg2[i][1],bnreg2[i][0]) for i in range(len(bnreg2))]
    bdpc2 = np.array(bdpc2)
    end = time.time()
    print('For parallel of bdreg, elapsed time is ' + str(end - start) + 'seconds...')

    return bdpc2, VamModel
