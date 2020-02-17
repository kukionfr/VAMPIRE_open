# built-in libraries
import pickle
from datetime import datetime
# external libraries
# my wrapper
from collect_selected_bstack import *
from update_csv import *
# my core
from bdreg import *
from pca_bdreg import *
from clusterSM import *


def mainbody(build_model, csv, entries, outpth=None, clnum=None, progress_bar=None):
    print('## main.py')
    progress = 50
    realtimedate = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    N = int(entries['Number of coordinates'].get())
    if build_model:
        bstack= collect_seleced_bstack(csv, build_model, entries)
        vampire_model = {
            "N": [],
            "bdrn": [],
            "mdd": [],
            "pc": [],
            "clnum": [],
            "pcnum": [],
            "mincms": [],
            "testmean": [],
            "teststd": [],
            "boxcoxlambda": [],
            "C": [],
            "Z": []
        }
        bdpc, vampire_model = bdreg(bstack[0], N, vampire_model, build_model)
        progress_bar["value"] = progress + 15
        progress_bar.update()
        score, vampire_model = pca_bdreg(bdpc, vampire_model, build_model)
        progress_bar["value"] = progress + 20
        progress_bar.update()
        pcnum = None # none is 20 by default
        IDX, IDX_dist, vampire_model= clusterSM(outpth, score, bdpc, clnum, pcnum, vampire_model, build_model, None,None, entries)
        progress_bar["value"] = progress + 25
        progress_bar.update()
        modelname = entries['Model name'].get()
        if os.path.exists(os.path.join(*[outpth, modelname, modelname+'.pickle'])):
            f = open(os.path.join(*[outpth, modelname, modelname+'_'+realtimedate+'.pickle']), 'wb')
        else:
            f = open(os.path.join(*[outpth, modelname, modelname+'.pickle']), 'wb')
        pickle.dump(vampire_model, f)
        f.close()

    else:
        UI = pd.read_csv(csv)
        setpaths = UI['set location']
        tag = UI['tag']
        condition = UI['condition']
        setID = UI['set ID'].astype('str')
        for setidx, setpath in enumerate(setpaths):
            pickles = [_ for _ in os.listdir(setpath) if _.lower().endswith('pickle')]
            bdstack = [pd.read_pickle(os.path.join(setpath, pkl)) for pkl in pickles if tag[setidx] in pkl]
            bdstacks = pd.concat(bdstack, ignore_index=True)
            progress_bar["value"] = 10
            progress_bar.update()
            try:
                f = open(entries['Model to apply'].get(), 'rb')
            except:
                entries['Status'].delete(0, END)  # global name END is not defined
                entries['Status'].insert(0, 'the model does not exist. please replace model name to the one you built')
            vampire_model = pickle.load(f)
            N = vampire_model['N']
            bdpc, vampire_model = bdreg(bdstacks[0], N, vampire_model, build_model)
            score, vampire_model = pca_bdreg(bdpc, vampire_model, build_model)
            clnum = vampire_model['clnum']
            pcnum = vampire_model['pcnum']
            IDX, IDX_dist, vampire_model = clusterSM(outpth,score,bdpc,clnum, pcnum, vampire_model,build_model,condition[setidx],setID[setidx],entries)
            tag = UI['tag'][setidx]
            update_csv(IDX, IDX_dist, tag, setpath)
            progress_bar["value"] = progress + 100 * (setidx + 1) / len(setpaths)
            progress_bar.update()
        entries['Status'].delete(0, END)
        entries['Status'].insert(0, 'applied the model')

