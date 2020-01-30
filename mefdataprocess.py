import pickle
from scipy.io import loadmat
import pandas as pd
# loc = r'\\motherserverdw\Lab Members\Kyu\Nature Protocol Submission Folder\VAMPIRE package\Supplementary files\Example segmented images\AG04054_Age29\c1_boundary_coordinate_stack.pickle'
# f = open(loc, 'rb')
# bdexample = pickle.load(f)
# print(bdexample)
mat = loadmat(r'C:\Users\kuki\Downloads\New Folder\kera_new\kera_boundary.mat')['bd']
df = pd.DataFrame(mat)
#mat2 = loadmat(r'\\babyserverdw4\Pei-Hsun Wu\digital pathology image data\Skin Tissue - Kyu\MAT\vampiredata\rest_18.mat')['restbd']
#df2 = pd.DataFrame(mat2)
#df3 = pd.concat([df,df2],ignore_index=True)
# print(df)
df.to_pickle(r'C:\Users\kuki\Downloads\New Folder\kera_new\kera_new.pickle')