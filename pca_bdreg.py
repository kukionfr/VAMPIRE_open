# built-in libraries
from __future__ import division
from copy import deepcopy
import time
import os
# external libraries
import numpy as np
from sklearn.decomposition import FastICA, PCA
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
        # latent is not useful
        pc, score, latent = PCA_custom(test)
        # score = np.dot(test, pc)

        dstt=r'\\motherserverdw\Lab Members\Kyu\Nature Protocol Submission Folder\Micropattern Validation'
        # np.save(os.path.join(dstt, 'pc.npy'), pc)
        # np.save(os.path.join(dstt, 'score.npy'), score)
        # pca = PCA(n_components=100)
        # pca.fit(test)
        # exvar = pca.explained_variance_ratio_
        # singvar = pca.singular_values_
        # activate below two lines to use ICA instead of PCA
        # transformer = FastICA(n_components=100, random_state=0)
        # score = transformer.fit_transform(test)
        # np.save(os.path.join(dstt, 'icaX.npy'), X_transformed)
        # time.sleep(1000)
    else:
        latent = VamModel['latent']
        pc = VamModel['pc']
        score = np.dot(test, pc)

    mdd = mmx[0]
    sdd = smx[0]

    VamModel['mdd'] = mdd
    VamModel['sdd'] = sdd
    VamModel['pc'] = pc
    VamModel['latent'] = latent
    end = time.time()
    print('For PCA bdreg, elapsed time is ' + str(end - start) + 'seconds...')
    return pc, score, latent, VamModel

