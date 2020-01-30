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


def mainbody(build_model, csv, entries, modelname=None, clnum=None, progress_bar=None):
    print('## main.py')
    progress = 50
    realtimedate = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    N = int(entries['Number of coordinates'].get())
    if build_model:
        figdst = os.path.join(os.path.dirname(csv), modelname)
        ch1, ch2 = collect_seleced_bstack(csv, build_model, entries)
        vampire_model = {
            "N": [],
            "bdrn": [],
            "mdd": [],
            "sdd": [],
            "pc": [],
            "latent": [],
            "clnum": [],
            "pcnum": [],
            "mincms": [],
            "testmean": [],
            "teststd": [],
            "boxcoxlambda": [],
            "C": [],
            "Z": []
        }
        vampire_model_ch2 = deepcopy(vampire_model)
        ch = 'ch1'
        bdpc, bnreg, sc, vampire_model = bdreg(ch1[0], N, vampire_model, build_model)
        progress_bar["value"] = progress + 15
        progress_bar.update()
        pc, score, latent, vampire_model = pca_bdreg(bdpc, vampire_model, build_model)
        pcnum = None
        progress_bar["value"] = progress + 20
        progress_bar.update()
        IDX, IDX_dist, bdsubtype, C, vampire_model, height, inertia = cluster_main(csv, modelname, score, pc, bdpc, clnum,
                                                                              pcnum, vampire_model, build_model, ch, None,
                                                                              entries)
        progress_bar["value"] = progress + 25
        progress_bar.update()
        if os.path.exists(os.path.join(*[figdst, 'Model for ' + ch, modelname + '_ch1.pickle'])):
            f = open(os.path.join(figdst, modelname + realtimedate + '_ch1.pickle'), 'wb')
            pickle.dump(vampire_model, f)
            f.close()
        else:
            f = open(os.path.join(*[figdst, 'Model for ' + ch, modelname + '_ch1.pickle']), 'wb')
            pickle.dump(vampire_model, f)
            f.close()
        ch = 'ch2'
        bdpc, bnreg, sc, vampire_model_ch2 = bdreg(ch2[0], N, vampire_model_ch2, build_model)
        progress_bar["value"] = progress + 35
        progress_bar.update()
        pc, score, latent, vampire_model_ch2 = pca_bdreg(bdpc, vampire_model_ch2, build_model)
        progress_bar["value"] = progress + 40
        progress_bar.update()
        pcnum = None
        IDX, IDX_dist, bdsubtype, C, vampire_model_ch2, height, intertia = cluster_main(csv, modelname, score, pc, bdpc,
                                                                                   clnum, pcnum, vampire_model_ch2,
                                                                                   build_model, ch, None, entries)
        progress_bar["value"] = progress + 45
        progress_bar.update()
        if os.path.exists(os.path.join(*[figdst, 'Model for ' + ch, modelname + '_ch2.pickle'])):
            f = open(os.path.join(*[figdst, 'Model for ' + ch, modelname + realtimedate + '_ch2.pickle']), 'wb')
            pickle.dump(vampire_model_ch2, f)
            f.close()
        else:
            f = open(os.path.join(*[figdst, 'Model for ' + ch, modelname + '_ch2.pickle']), 'wb')
            pickle.dump(vampire_model_ch2, f)
            f.close()

    else:
        model_list = []
        for root, dirs, files in os.walk(modelname, topdown=True):
            for name in files:
                if name.endswith('pickle'): model_list.append(os.path.join(root, name))

        UI = pd.read_csv(csv)
        setpaths = UI['set location']
        ch1ui = UI['ch1']
        ch2ui = UI['ch2']
        condition = UI['condition']
        for setidx, setpath in enumerate(setpaths):
            pickles = [_ for _ in os.listdir(setpath) if _.lower().endswith('pickle')]

            c1_stack = [pd.read_pickle(os.path.join(setpath, pkl)) for pkl in pickles if ch1ui[setidx] in pkl]
            c2_stack = [pd.read_pickle(os.path.join(setpath, pkl)) for pkl in pickles if ch2ui[setidx] in pkl]

            ch1 = pd.concat(c1_stack, ignore_index=True)
            ch2 = pd.concat(c2_stack, ignore_index=True)

            progress_bar["value"] = 10
            progress_bar.update()

            try:
                f = open(model_list[0], 'rb')
            except:
                entries['Status'].delete(0, END)  # global name END is not defined
                entries['Status'].insert(0, 'the model does not exist. please replace model name to the one you built')
            vampire_model = pickle.load(f)

            N = vampire_model['N']
            ch = 'ch1'
            bdpc_new, bnreg_new, sc_new, vampire_model = bdreg(ch1[0], N, vampire_model, build_model)
            pc_new, score_new, latent_new, vampire_model = pca_bdreg(bdpc_new, vampire_model, build_model)
            clnum = vampire_model['clnum']
            pcnum = vampire_model['pcnum']
            # pc_new goes in for sake of placing, but pc from the model is used in cluster_main
            resultdst = os.path.join(os.path.dirname(os.path.dirname(model_list[0])), 'Result for ' + ch)
            IDX_ch1, IDX_dist_ch1, bdsubtype_new, C_new, vampire_model, height1, inertia1 = cluster_main(csv, resultdst,
                                                                                                    score_new, pc_new,
                                                                                                    bdpc_new, clnum,
                                                                                                    pcnum, vampire_model,
                                                                                                    build_model, ch,
                                                                                                    condition[setidx],
                                                                                                    entries
                                                                                                    )
            update_csv(IDX_ch1, IDX_dist_ch1, UI, ch, setpath)
            try:
                f = open(model_list[1], 'rb')
            except:
                entries['Status'].delete(0, END)
                entries['Status'].insert(0, 'Please use exact name of the model you have built to apply here')
            vampire_model = pickle.load(f)
            N = vampire_model['N']
            ch = 'ch2'
            bdpc_new, bnreg_new, sc_new, vampire_model = bdreg(ch2[0], N, vampire_model, build_model)
            pc_new, score_new, latent_new, vampire_model = pca_bdreg(bdpc_new, vampire_model, build_model)
            clnum = vampire_model['clnum']
            pcnum = vampire_model['pcnum']
            # pc_new goes in for sake of placing, but pc from the model is used in cluster_main
            resultdst = os.path.join(os.path.dirname(os.path.dirname(model_list[0])), 'Result for ' + ch)
            IDX_ch2, IDX_dist_ch2, bdsubtype_new, C_new, vampire_model, height2, inertia2 = cluster_main(csv, resultdst,
                                                                                                    score_new, pc_new,
                                                                                                    bdpc_new, clnum,
                                                                                                    pcnum, vampire_model,
                                                                                                    build_model, ch,
                                                                                                    condition[setidx],
                                                                                                    entries
                                                                                                    )
            update_csv(IDX_ch2, IDX_dist_ch2, UI, ch, setpath)
            progress_bar["value"] = progress + 100 * (setidx + 1) / len(setpaths)
            progress_bar.update()
        entries['Status'].delete(0, END)
        entries['Status'].insert(0, 'applied the model')

