#!/usr/bin/env python

# internal libraries
from __future__ import division
from copy import deepcopy
# from inspect import getargspec
import time
import os
from tkinter import END
from time import sleep
# external libraries
import numpy as np
from scipy import stats, cluster, spatial
from sklearn.cluster import KMeans, DBSCAN
from sklearn import preprocessing, metrics
import matplotlib as mpl
mpl.use("TkAgg")
from matplotlib import pyplot as plt


def clusterSM(csv, modelname, score, pc, bdpc, clnum=None, pcnum=None, VamModel=None, BuildModel=None, ch=None,
              condition=None, entries=None):
    print('# clusterSM')
    csvdir = os.path.dirname(csv)
    figdst = os.path.join(csvdir, modelname)
    if BuildModel:
        figdst = os.path.join(figdst, 'Model for ' + ch)
    else:
        figdst = modelname
    if not os.path.exists(figdst):
        try:
            os.makedirs(figdst)
        except:
            entries['Status'].delete(0, END)
            entries['Status'].insert(0, 'Please fill in model name correctly')

    Nuu = int(round(len(bdpc.T[0])))
    Nbb = int(round(len(bdpc[0]) / 2))
    mmx = np.dot(np.ones([Nuu, 1]), np.mean([bdpc], axis=1))
    smx = np.ones(bdpc.shape)
    mdd = mmx[0]
    sdd = smx[0]
    NN = 10

    if clnum is None:
        clnum = 15
    if pcnum is None:
        pcnum = 20

    if BuildModel:
        VamModel['clnum'] = clnum
        VamModel['pcnum'] = pcnum
    else:
        clnum = VamModel['clnum']
        pcnum = VamModel['pcnum']
        pc = VamModel['pc']

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
            #####################
            VamModel['boxcoxlambda'][k] = maxlog
            VamModel['testmean'][k] = np.mean(test)
            VamModel['teststd'][k] = np.std(test)
            #####################
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
        unique_labels2 = set(IDX)
        IDX_dist = np.amin(D, axis=1)
        inertia = [np.around(np.sum(np.square(IDX_dist[IDX == _])) / len(IDX_dist[IDX == _]), decimals=2) for _ in
                   range(clnum)]

        # eps = max dist btw two points in a neighbor; min_sample = min data points to consider it as a neighbor
        # n_jobs = -1 for all processors; algorithm = ball_tree, kd_tree, brute
        #db = DBSCAN(eps=0.4, min_samples=5, algorithm='auto', leaf_size=30, n_jobs=-1).fit(cmsn_Norm)
        # samples with labels that are in core
        #core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        #core_samples_mask[db.core_sample_indices_] = True
        #labels = db.labels_
        #n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        #n_noise = list(labels).count(-1)
        #print('Estimated number of clusters: %d' % n_clusters)
        #print('Estimated number of noise points: %d' % n_noise)
        # Black removed and is used for noise instead.
        #unique_labels = set(labels)
        #colors = [plt.cm.Spectral(each)
        #          for each in np.linspace(0, 1, len(unique_labels))]
        #colors2 = [plt.cm.Spectral(each2)
        #          for each2 in np.linspace(0, 1, len(unique_labels2))]
        #fig1, ax1 = plt.subplots()
        #fig2, ax2 = plt.subplots()
        #for k, col in zip(unique_labels, colors):
        #    if k == -1:
                # Black used for noise.
        #        col = [0, 0, 0, 1]

         #   class_member_mask = (labels == k)

         #   xy = cmsn_Norm[class_member_mask & core_samples_mask]
         #   ax1.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
         #            markeredgecolor='k', markersize=14)

        #    xy = cmsn_Norm[class_member_mask & ~core_samples_mask]
         #   ax1.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
        #             markeredgecolor='k', markersize=6)

        #for j, col in zip(unique_labels2,colors2):
        #    class_member_mask2 = (IDX == j)
        #    xy = cmsn_Norm[class_member_mask2]
        #    ax2.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
        #             markeredgecolor='k', markersize=6)
        #ax1.set_title('BDscan - Estimated number of clusters: %d' % n_clusters)
        #ax2.set_title('K-means - Set number of clusters: %d' % clnum)

        #db2 = DBSCAN(eps=0.5, min_samples=5, algorithm='auto', leaf_size=30, n_jobs=-1).fit(cmsn_Norm)
        #core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        #core_samples_mask[db.core_sample_indices_] = True
        #labels = db.labels_
        #n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        #n_noise = list(labels).count(-1)
        #print('Estimated number of clusters: %d' % n_clusters)
        #print('Estimated number of noise points: %d' % n_noise)
        #unique_labels = set(labels)
        #colors = [plt.cm.Spectral(each)
        #          for each in np.linspace(0, 1, len(unique_labels))]
        #fig3, ax3 = plt.subplots()
        #for k2, col2 in zip(unique_labels, colors):
        #    if k2 == -1:
        #        # Black used for noise.
        #        col2 = [0, 0, 0, 1]

         #   class_member_mask = (labels == k)

         #   xy = cmsn_Norm[class_member_mask & core_samples_mask]
         #   ax3.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col2),
         #            markeredgecolor='k', markersize=14)

   #xy = cmsn_Norm[class_member_mask & ~core_samples_mask]
    #        ax3.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col2),
     #                markeredgecolor='k', markersize=6)
     #  plt.show()
        #sleep(1000)
    else:
        if isinstance(clnum, str):
            clnum = int(clnum)
        C = VamModel['C']
        D = spatial.distance.cdist(cmsn, C, metric='euclidean')
        # why amin? D shows list of distance to cluster centers.
        IDX = np.argmin(D, axis=1)
        IDX_dist = np.around(np.amin(D, axis=1), decimals=2)
        inertia = [np.around(np.sum(np.square(IDX_dist[IDX == _])) / len(IDX_dist[IDX == _]), decimals=2) for _ in
                   range(clnum)]

    offx, offy = np.meshgrid(range(clnum), [0])
    offx = np.multiply(offx, 1) + 1
    offx = offx[0] * 1 - 0.5
    offy = np.subtract(np.multiply(offy, 1), 1.5) + 1
    offy = offy[0]
    # define normalized colormap
    cmap = plt.cm.jet
    vmax = int(clnum * 10)
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
    cid = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    clshape = np.zeros([clnum, len(cms00[0])])
    clshapesdv = deepcopy(clshape)
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
    bdsubtype = np.empty(
        (int(max(IDX) + 1), 2, Nbb + 1))  # need more specific preallocation: 2 for x and y, Nbb+1 for len(x)
    fig922, ax922 = plt.subplots(figsize=(17, 2))
    fig291, ax291 = plt.subplots(figsize=(6, 3))
    for kss in range(int(max(IDX)) + 1):
        c88 = IDXsort == kss
        # clshape[kss] = np.mean(cms00[c88], axis=0)
        # clshapesdv[kss] = np.std(cms00[c88], axis=0)
        # pnn = np.zeros(len(pc.T[0]))
        # for kev in range(len(cms00[0])):
        #     pnn = np.add(pnn, np.multiply(pc.T[kev], clshape[kss, kev]))
        #     pnnlb = np.add(pnn, np.multiply(np.multiply(pc.T[kev], -2), clshapesdv[kss, kev]))
        #     pnnhb = np.add(pnn, np.multiply(np.multiply(pc.T[kev], 2), clshapesdv[kss, kev]))
        # pnn = np.multiply(pnn, sdd) + mdd
        # pnnlb = np.multiply(pnnlb, sdd) + mdd
        # pnnhb = np.multiply(pnnhb, sdd) + mdd  # pnn,pnnlb&hb are all randomized
        # xx = pnn[0:Nbb]
        # yy = pnn[Nbb:]
        # xx = np.append(xx, xx[0])
        # yy = np.append(yy, yy[0])
        fss = 4
        # # this plots what? figeach together?
        # ax291.plot((xx / fss + offx.T[kss]) * 10, (yy / fss + offy.T[kss]) * 10, '-', color=cid.to_rgba(kss),
        #            linewidth=5)  # this is not plotted in matlab as well
        # plt.axis('equal')
        #
        # bdsubtype[kss][0] = xx / fss
        # bdsubtype[kss][1] = yy / fss
        bdpcs = bdpc[c88]
        mbd = np.mean(bdpcs, axis=0)
        bdNUM = int(round(len(mbd) / 2))
        bdst = np.vstack((bdst, mbd))
        xaxis = np.add(np.divide(np.append(mbd[0:bdNUM], mbd[0]), fss), offx[kss]) * 10
        yaxis = np.add(np.divide(np.append(mbd[bdNUM:], mbd[bdNUM]), fss), offy[kss]) * 10
        plt.clf()

        ax289.plot(xaxis, yaxis, '-', linewidth=2)  # this is the shape of the dendrogram
        # plt.gca().invert_yaxis()
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
            ax922.axis('off')
    bdst = bdst[1:]
    if BuildModel:
        fig922.savefig(os.path.join(figdst, "Registered objects " + ch + ".png"), format='png', transparent=True)
        # fig922.savefig(os.path.join(figdst,"Registered objects "+ch+".svg"),format='svg',transparent=True)
        fig289.savefig(os.path.join(figdst, "Shape mode dendrogram " + ch + ".png"), format='png', transparent=True)
        # fig289.savefig(os.path.join(figdst,"Shape mode dendrogram "+ch+".svg"),format='svg',transparent=True)

    IDX = IDX + 1
    n, bins, patches = plt.hist(IDX, bins=range(clnum + 2)[1:])
    fig22, ax22 = plt.subplots(figsize=(10, 5))
    n = np.divide(n, np.sum(n))
    n = np.multiply(n, 100)
    n = np.around(n, 2)
    height = n
    ax22.bar(x=(np.delete(bins, 0) - 1) / 2, height=height, width=0.4, align='center', color=(0.2, 0.4, 0.6, 0.6),
             edgecolor='black')

    ax22.set_ylabel('Abundance %')
    ax22.set_xlabel('Shape mode')
    # only for paper
    ax22.set_ylim([0,30])
    ax22.set_title('Shape mode distribution (N=' + str(len(IDX_dist)) + ')')
    bartick = map(str, np.arange(int(np.max(IDX) + 1))[1:])
    ax22.set_xticks((np.arange(np.max(IDX) + 1) / 2)[1:])
    ax22.set_xticklabels(tuple(bartick))

    for i, v in enumerate(height):
        ax22.text((i - 0.3 + 1) / 2, v + 0.25, str(np.around(v, decimals=1)), color='black', fontweight='bold')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax22.spines[axis].set_linewidth(3)

    if not BuildModel:
        fig22.savefig(os.path.join(figdst, 'shape mode distribution_' + ch + '_' + condition + '.png'), format='png',
                      transparent=True)
        # fig22.savefig(os.path.join(figdst,'shape mode distribution_'+ch+'_'+condition+'.svg'),format='svg',transparent=True)

    plt.close('all')
    return IDX, IDX_dist, bdsubtype, C, VamModel, height, inertia


def cluster_main(csv, modelname, score, pc, bdpc, clnum=None, pcnum=None, VamModel=None, BuildModel=None,
                 cellornuc=None, condition=None, entries=None):
    print('## clusterSM.py')
    start = time.time()
    if not isinstance(condition, str):
        condition = str(condition)
    IDX, IDX_dist, bdsubtype, C, VamModel, height, inertia = clusterSM(csv, modelname, score, pc, bdpc, clnum, pcnum,
                                                                       VamModel, BuildModel, cellornuc, condition,
                                                                       entries)
    end = time.time()
    print('For cluster, elapsed time is ' + str(end - start) + 'seconds...')
    return IDX, IDX_dist, bdsubtype, C, VamModel, height, inertia
