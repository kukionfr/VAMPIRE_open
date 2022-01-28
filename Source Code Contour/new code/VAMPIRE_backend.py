import cv2
import time
import os
from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib as mpl
from copy import deepcopy
import numpy as np
from scipy import interpolate
import math
from scipy import stats, cluster, spatial, special
from sklearn.cluster import KMeans
from sklearn import preprocessing
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd

def update_csv(idx, fit, tag, setpath, **kwargs):
    print('## update_csv.py')
    datasheet = 'VAMPIRE datasheet ' + tag + '.csv'
    goodness = kwargs['goodness'].transpose()
    if os.path.exists(os.path.join(setpath, datasheet)):
        obj_ledger = pd.read_csv(os.path.join(setpath, datasheet))
        obj_ledger['Shape mode'] = pd.Series(idx)
        obj_ledger['Distance from cluster center'] = pd.Series(fit)
        for idx,column in enumerate(goodness):
            obj_ledger['probability of shape mode '+str(idx)] = column
        obj_ledger.to_csv(os.path.join(setpath, datasheet), index=False)
    else:
        d = {'Shape mode': pd.Series(idx), 'Distance from cluster center': pd.Series(fit)}
        for idx,column in enumerate(goodness):
            d['probability of shape mode '+str(idx)] = column
        obj_ledger = pd.DataFrame(data=d)
        obj_ledger.to_csv(os.path.join(setpath, datasheet), index=False)

def cntarea(cnt):
    cnt = np.array(cnt)
    area = cv2.contourArea(cnt)
    return area

def cntAR(cnt):
    cnt = np.array(cnt)
    #Orientation, Aspect_ratio
    (x,y),(MA,ma),orientation = cv2.fitEllipse(cnt)
    aspect_ratio = MA/ma
    return aspect_ratio
def cntExtent(cnt):
    cnt = np.array(cnt)
    x,y,w,h = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)
    rect_area = w*h
    extent = float(area)/rect_area
    return extent
def cntEquiDia(cnt):
    cnt = np.array(cnt)
    #Equi Diameter
    area = cv2.contourArea(cnt)
    equi_diameter = np.sqrt(4*area/np.pi)
    return equi_diameter
def cntsol(cnt):
    cnt = np.array(cnt)
    #Solidity
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area)/hull_area
    return solidity
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

    start = time.time()
    bnreg_a = deepcopy(B)
    bnreg2 = Parallel(n_jobs=num_cores)(delayed(reg_bd3)(bnreg_a.loc[kk], bdrn) for kk in range(kll))
    bdpc2 = [np.append(bnreg2[i][1],bnreg2[i][0]) for i in range(len(bnreg2))]
    bdpc2 = np.array(bdpc2)
    end = time.time()
    print('For parallel of bdreg, elapsed time is ' + str(end - start) + 'seconds...')
    return bdpc2, VamModel

def reg_bd_svd(bd0=None):
    xc = np.mean(bd0[1])
    yc = np.mean(bd0[0])
    bd0 = np.append([bd0[0]] - yc, [bd0[1]] - xc, axis=0)
    xi = bd0[1]
    yi = bd0[0]
    s = np.sqrt((sum(np.power(xi, 2)) + sum(np.power(yi, 2))) / len(xi))
    xi = xi / s
    yi = yi / s
    xiyi = np.append([xi], [yi], axis=0).transpose()
    u, S, rm = np.linalg.svd(xiyi, full_matrices=True)
    # this is a quick fix
    if np.isnan(rm).any():
        rm[rm!=1]=1
    xynew = np.dot(xiyi, rm.transpose())
    xynew = xynew.transpose()
    yc = xynew[1].mean()
    xc = xynew[0].mean()
    xon = xynew[0] - xc
    yon = xynew[1] - yc
    theta = np.empty(len(yon))
    theta[:] = np.nan
    for i in range(len(yon)):
        theta[i] = math.atan2(yon[i], xon[i])
    cc = np.argwhere(abs(theta) == min(abs(theta)))
    cc = cc[0]
    cc = cc[0]
    ccid = np.append(range(cc, len(xon)), range(0, cc))
    ccid = ccid.astype(int)
    xon = xon[ccid]
    yon = yon[ccid]
    theta = theta[ccid]
    if theta[4] - theta[0] < 0:
        xon = np.append(xon[-1:0:-1], xon[0])
        yon = np.append(yon[-1:0:-1], yon[0])
    regbd = np.append([yon], [xon], axis=0)
    return regbd, s

def bd_resample(bd=None, rsnum=None):
    bd = np.array(bd)
    x = bd.T[1]
    y = bd.T[0]
    if x[-1] != x[0] and y[-1] != y[0]:
        x = np.append(x, x[0])
        y = np.append(y, y[0])
    sd = np.sqrt(np.power(x[1:]-x[0:-1], 2) + np.power(y[1:]-y[0:-1], 2))
    sd = np.append([1], sd)
    sid = np.cumsum(sd)
    ss = np.linspace(1, sid[-1], rsnum + 1)
    ss = ss[0:-1]
    splinerone = interpolate.splrep(sid, x, s=0)
    sx = interpolate.splev(ss, splinerone, der=0)
    splinertwo = interpolate.splrep(sid, y, s=0)
    sy = interpolate.splev(ss, splinertwo, der=0)
    bdrs = np.append([sy], [sx], axis=0)
    return bdrs

def reg_bd3(bd0=None, bdr0=None):
    xc = np.sum(np.dot(bd0[1], abs(bd0[0]))) / np.sum(abs(bd0[0]))
    yc = np.sum(np.dot(bd0[0], abs(bd0[1]))) / np.sum(abs(bd0[1]))

    bd0 = np.append([bd0[0] - yc], [bd0[1] - xc], axis=0)
    bd = bd0
    bdr = bdr0

    xc = np.sum(np.dot(bdr[1], abs(bdr[0]))) / np.sum(abs(bdr[0]))
    yc = np.sum(np.dot(bdr[0], abs(bdr[1]))) / np.sum(abs(bdr[1]))

    bdr = np.append([bdr[0] - yc], [bdr[1] - xc], axis=0)
    temp = deepcopy(bdr[1])
    bdr[1] = bdr[0]
    bdr[0] = temp
    temp = deepcopy(bd[1])
    bd[1] = bd[0]
    bd[0] = temp
    N = len(bd[0])
    costold = np.mean(sum(sum(np.power((bdr - bd), 2))))
    bdout = deepcopy(bd)
    # print('regbd3')
    for k in range(1, N + 1):
        idk = np.append(range(k, N + 1), range(1, k))
        bdt = np.empty([len(idk), 2])
        bdt[:] = np.nan
        for i in range(len(bd.transpose())):
            ind = int(idk[i] - 1)
            bdt[i] = bd.transpose()[ind]
        temp = np.dot(bdr, bdt)
        u, _, v = np.linalg.svd(temp)
        v = v.T
        q = np.dot(v, u.transpose())
        bdtemp = np.dot(bdt, q)
        costnew = np.mean(sum(sum(np.power((bdr.transpose() - bdtemp), 2))))
        if costnew < costold:
            bdout = deepcopy(bdtemp)
            costold = deepcopy(costnew)

    regbd = deepcopy(bdout.T)
    regbd[:] = np.nan
    regbd[0] = deepcopy(bdout.T[1])
    regbd[1] = deepcopy(bdout.T[0])
    return regbd
def PCA_custom(data, dims_rescaled_data=100):
    """
    returns: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """
    import numpy as np
    from scipy import linalg as la
    # m, n = data.shape
    # mean center the datam
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    r = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since r is symmetric,
    # the performance gain is substantial
    evals, evecs = la.eigh(r)
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return evecs, np.dot(evecs.T, data.T).T, evals

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
def clusterSM(outpth, score, bdpc, clnum, pcnum=None, VamModel=None, BuildModel=None,
              condition=None,setID=None,modelname=None):
    realtimedate = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    start = time.time()
    print('# clusterSM')
    if not isinstance(condition, str):
        condition = str(condition)
    if BuildModel:
        figdst = os.path.join(*[outpth,modelname, 'Example model figures'])
    else:
        figdst = os.path.join(outpth, 'Result based on ' + modelname)
    if not os.path.exists(figdst):
        try:
            os.makedirs(figdst)
        except:
            print('plz choose right folder')
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

        kmeans = KMeans(n_clusters=clnum, init='k-means++', n_init=3, max_iter=300, random_state=9).fit(
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
    goodness = special.softmax(1-D,axis=1)
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

