# built-in libraries

# external libraries
import numpy as np
from scipy import interpolate


def bd_resample(bd=None, rsnum=None):
    x = bd.transpose()[1]
    y = bd.transpose()[0]
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