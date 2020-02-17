# built-in libraries
from __future__ import division
from copy import deepcopy
import time
# external libraries
import numpy as np
# my files
from PCA_custom import PCA_custom


def pca_bdreg(bdpc, VamModel, BuildModel):
    print('## pca_bdreg.py')
    start = time.time()
    Nuu = int(round(len(bdpc.T[0])))
    bdpct = deepcopy(bdpc)

    if BuildModel:
        mmx = np.ones((Nuu, 1)) * np.mean(bdpct, axis=0)
    else:
        mmx = np.ones((Nuu, 1)) * VamModel['mdd']
    smx = np.ones(bdpct.shape)
    test = np.divide((bdpct - mmx), smx)
    if BuildModel:
        # latent is not used later
        pc, score, latent = PCA_custom(test)
        VamModel['pc'] = pc
    else:
        pc = VamModel['pc']
        score = np.dot(test, pc)
    mdd = mmx[0]
    VamModel['mdd'] = mdd
    end = time.time()
    print('For PCA bdreg, elapsed time is ' + str(end - start) + 'seconds...')
    return score, VamModel

