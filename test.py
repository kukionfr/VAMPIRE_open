import os
import numpy as np
dstt= r'C:\Users\kuki\Desktop\sell'
bnreg = np.load(os.path.join(dstt, 'bnreg.npy'),allow_pickle=True)
bnreg2 = np.load(os.path.join(dstt, 'bnreg2.npy'),allow_pickle=True)
bdpc = np.load(os.path.join(dstt, 'bdpc.npy'),allow_pickle=True)
bdpc2 = np.load(os.path.join(dstt, 'bdpc2.npy'),allow_pickle=True)
a = bnreg[0][1]
b = bnreg[0][0]
aa = bnreg2[0][1]
bb = bnreg2[0][0]
ar = [len(_) for _ in bnreg]
uni = np.unique(ar)
