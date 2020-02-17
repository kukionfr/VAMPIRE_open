import pickle
from scipy.io import loadmat
import pandas as pd
import os
# loc = r'\\motherserverdw\Lab Members\Kyu\Nature Protocol Submission Folder\VAMPIRE package\Supplementary files\Example segmented images\AG04054_Age29\c1_boundary_coordinate_stack.pickle'
# f = open(loc, 'rb')
# bdexample = pickle.load(f)
# print(bdexample)
pth = r'\\motherserverdw\Lab Members\Kyu\Active\PDAC cell detection project\immune cell detected'
matlist = [os.path.join(pth,_) for _ in os.listdir(pth) if _.endswith('mat')]
for mat in matlist:
    filename, ext = os.path.splitext(os.path.basename(mat))
    if os.path.exists(os.path.join(*[pth,filename+'.pickle'])): continue
    matbd = loadmat(mat)
    matbd = matbd['bd']
    df = pd.DataFrame(matbd)
    df.to_pickle(os.path.join(*[pth,filename+'.pickle']))
    print('hi')
#mat2 = loadmat(r'\\babyserverdw4\Pei-Hsun Wu\digital pathology image data\Skin Tissue - Kyu\MAT\vampiredata\rest_18.mat')['restbd']
#df2 = pd.DataFrame(mat2)
#df3 = pd.concat([df,df2],ignore_index=True)
# print(df)
