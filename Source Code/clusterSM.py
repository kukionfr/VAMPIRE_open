#!/usr/bin/env python

# internal libraries
from __future__ import division
from copy import deepcopy
import time
import os
from tkinter import END
from datetime import datetime
# external libraries
import numpy as np
from scipy import stats, cluster, spatial, special
from sklearn.cluster import KMeans
from sklearn import preprocessing
import matplotlib as mpl
mpl.use("TkAgg")
from matplotlib import pyplot as plt


def clusterSM(outpth, score, bdpc, clnum, pcnum=None, VamModel=None, BuildModel=None,
              condition=None,setID=None, entries=None):
    realtimedate = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    start = time.time()
    print('# clusterSM')
    if not isinstance(condition, str):
        condition = str(condition)
    if BuildModel:
        figdst = os.path.join(*[outpth, entries['Model name'].get(), 'Example model figures'])
    else:
        figdst = os.path.join(outpth, 'Result based on ' + os.path.splitext(os.path.basename(entries['Model to apply'].get()))[0])
    if not os.path.exists(figdst):
        try:
            os.makedirs(figdst)
        except:
            entries['Status'].delete(0, END)
            entries['Status'].insert(0, 'Please choose the right folder')
    NN = 10

    if pcnum is None:
        pcnum = 20

    if BuildModel:
        VamModel['clnum'] = clnum
        VamModel['pcnum'] = pcnum
    else:
        clnum = VamModel['clnum']
        pcnum = VamModel['pcnum']

    cms00 = score[:, 0:pcnum]
    cms = deepcopy(cms00)

    if BuildModel:
        mincms = np.amin(cms, axis=0)
        VamModel['mincms'] = mincms
        VamModel['boxcoxlambda'] = np.zeros(len(cms.T))
        VamModel['testmean'] = np.zeros(len(cms.T))
        VamModel['teststd'] = np.zeros(len(cms.T))
    else:
        mincms = VamModel['mincms']

    for k in range(len(cms.T)):
        test = cms.T[k]
        test = test - mincms[k] + 1
        if BuildModel:
            test[test < 0] = 0.000000000001
            test, maxlog = stats.boxcox(test)
            test = np.asarray(test)
            VamModel['boxcoxlambda'][k] = maxlog
            VamModel['testmean'][k] = np.mean(test)
            VamModel['teststd'][k] = np.std(test)
            cms.T[k] = (test - np.mean(test)) / np.std(test)
        else:
            test[test < 0] = 0.000000000001
            test = stats.boxcox(test, VamModel['boxcoxlambda'][k])
            cms.T[k] = (test - VamModel['testmean'][k]) / VamModel['teststd'][k]

    cmsn = deepcopy(cms)

    if BuildModel:
        cmsn_Norm = preprocessing.normalize(cmsn)
        if isinstance(clnum, str):
            clnum = int(clnum)

        kmeans = KMeans(n_clusters=clnum, init='k-means++', n_init=3, max_iter=300,n_jobs=-1).fit(
            cmsn_Norm)  # init is plus,but orginally cluster, not available in sklearn
        C = kmeans.cluster_centers_
        VamModel['C'] = C
        D = spatial.distance.cdist(cmsn, C, metric='euclidean')
        IDX = np.argmin(D, axis=1)
        IDX_dist = np.amin(D, axis=1)
    else:
        if isinstance(clnum, str):
            clnum = int(clnum)
        C = VamModel['C']
        D = spatial.distance.cdist(cmsn, C, metric='euclidean')
        # why amin? D shows list of distance to cluster centers.
        IDX = np.argmin(D, axis=1)
        IDX_dist = np.around(np.amin(D, axis=1), decimals=2)
    goodness = special.softmax(D)
    offx, offy = np.meshgrid(range(clnum), [0])
    offx = np.multiply(offx, 1) + 1
    offx = offx[0] * 1 - 0.5
    offy = np.subtract(np.multiply(offy, 1), 1.5) + 1
    offy = offy[0]
    # define normalized colormap
    bdst0 = np.empty(len(bdpc.T))
    bdst = deepcopy(bdst0)
    for kss in range(clnum):
        c88 = IDX == kss
        bdpcs = bdpc[c88, :]
        mbd = np.mean(bdpcs, axis=0)
        bdst0 = np.vstack((bdst0, mbd))
    bdst0 = bdst0[1:]
    # dendrogram of the difference between different shape
    mpl.rcParams['lines.linewidth'] = 2
    if BuildModel:
        Y = spatial.distance.pdist(bdst0, 'euclidean')
        Z = cluster.hierarchy.linkage(Y, method='complete')  # 4th row is not in matlab
        Z[:, 2] = Z[:, 2] * 5  # multiply distance manually 10times to plot better.
        VamModel['Z'] = Z
    else:
        Z = VamModel['Z']
    cluster.hierarchy.set_link_color_palette(['k'])
    fig289, ax289 = plt.subplots(figsize=(6, 2), linewidth=2.0, frameon=False)
    plt.yticks([])
    R = cluster.hierarchy.dendrogram(Z, p=0, truncate_mode='mlab', orientation='bottom', ax=None,
                                     above_threshold_color='k')
    leaflabel = np.array(R['ivl'])
    dendidx = leaflabel
    cluster.hierarchy.set_link_color_palette(None)
    mpl.rcParams['lines.linewidth'] = 1
    plt.axis('equal')
    plt.axis('off')
    IDXsort = np.zeros(len(IDX))
    for kss in range(clnum):
        c88 = IDX == int(dendidx[kss])
        IDXsort[c88] = kss
    IDX = deepcopy(IDXsort)
    fig922, ax922 = plt.subplots(figsize=(17, 2))
    fig291, ax291 = plt.subplots(figsize=(6, 3))
    for kss in range(int(max(IDX)) + 1):
        c88 = IDXsort == kss
        fss = 4
        bdpcs = bdpc[c88]
        mbd = np.mean(bdpcs, axis=0)
        bdNUM = int(round(len(mbd) / 2))
        bdst = np.vstack((bdst, mbd))
        xaxis = np.add(np.divide(np.append(mbd[0:bdNUM], mbd[0]), fss), offx[kss]) * 10
        yaxis = np.add(np.divide(np.append(mbd[bdNUM:], mbd[bdNUM]), fss), offy[kss]) * 10
        plt.clf()

        ax289.plot(xaxis, yaxis, '-', linewidth=2)  # this is the shape of the dendrogram
        plt.axis('equal')
        plt.axis('off')

        sid = np.argsort(np.random.rand(sum(c88), 1), axis=0)
        if len(sid) < NN:
            enum = len(sid)
        else:
            enum = NN
        for knn in range(enum):
            x99 = bdpcs[sid[knn], np.append(range(bdNUM), 0)]
            y99 = bdpcs[sid[knn], np.append(np.arange(bdNUM, (bdNUM * 2), 1), bdNUM)]
            xax = np.add(np.divide(x99, fss), offx[kss])
            yax = np.add(np.divide(y99, fss), offy[kss])
            ax922.plot(xax, yax, 'r-', linewidth=1)
            ax922.axis('equal')
            ax922.axis('off')
    if BuildModel:
        ax922.set_ylim(ax922.get_ylim()[::-1])
        if os.path.exists(os.path.join(figdst, "Registered objects.png")):
            f1 = os.path.join(figdst, "Registered objects "+realtimedate+".png")
            f2 = os.path.join(figdst, "Shape mode dendrogram.png "+realtimedate+".png")
        else:
            f1 = os.path.join(figdst, "Registered objects.png")
            f2 = os.path.join(figdst, "Shape mode dendrogram.png")
        fig922.savefig(f1, format='png', transparent=True)
        fig289.savefig(f2, format='png', transparent=True)

    IDX = IDX + 1
    n, bins, patches = plt.hist(IDX, bins=range(clnum + 2)[1:])
    fig22, ax22 = plt.subplots(figsize=(10, 5))
    n = np.divide(n, np.sum(n))
    n = np.multiply(n, 100)
    n = np.around(n, 2)
    height = n
    ax22.bar(x=(np.delete(bins, 0) - 1) / 2, height=height, width=0.4, align='center', color=(0.2, 0.4, 0.6, 1),
             edgecolor='black')
    ax22.set_ylabel('Abundance %', fontsize=15, fontweight='bold')
    ax22.set_xlabel('Shape mode', fontsize=15, fontweight='bold')
    # only for paper
    ax22.set_ylim([0,np.max(height)+5])

    ax22.set_title('Shape mode distribution (N=' + str(len(IDX_dist)) + ')',fontsize=18, fontweight='bold')
    bartick = map(str, np.arange(int(np.max(IDX) + 1))[1:])
    ax22.set_xticks((np.arange(np.max(IDX) + 1) / 2)[1:])
    ax22.set_xticklabels(tuple(bartick), fontsize=13, fontweight='bold')
    ax22.yaxis.set_tick_params(labelsize=13)
    plt.setp(ax22.get_yticklabels(), fontweight="bold")
    for i, v in enumerate(height):
        ax22.text((i - 0.25 + 1) / 2, v + 0.25, str(np.around(v, decimals=1)), color='black', fontweight='bold', fontsize=13)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax22.spines[axis].set_linewidth(3)
    if not BuildModel:
        if os.path.exists(os.path.join(figdst, 'Shape mode distribution_'+ setID + '_' + condition + '.png')):
            f3 = os.path.join(figdst, 'Shape mode distribution_'+ setID + '_' + condition +'_'+realtimedate+'.png')
        else:
            f3 = os.path.join(figdst, 'Shape mode distribution_'+ setID + '_' + condition + '.png')
        fig22.savefig(f3, format='png', transparent=True)
    plt.close('all')
    end = time.time()
    print('For cluster, elapsed time is ' + str(end - start) + 'seconds...')
    return IDX, IDX_dist, VamModel, goodness


