# built-in libraries
import math
# external libraries
import numpy as np
from scipy import linalg


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
    u, S, rm = linalg.svd(xiyi)
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
