import os
import numpy as np
fol = r'\\motherserverdw\Lab Members\Kyu\Nature Protocol Submission Folder\Micropattern Validation'
npy = [os.path.join(fol,_) for _ in os.listdir(fol) if _.endswith('.npy')]
ica = np.load(npy[0])
pc = np.load(npy[1])
score = np.load(npy[2])

