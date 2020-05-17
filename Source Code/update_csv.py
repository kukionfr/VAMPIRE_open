# built-in libraries
import os
# external libraries
import pandas as pd


def update_csv(idx, fit, tag, setpath, **kwargs):
	print('## update_csv.py')
	datasheet = 'VAMPIRE datasheet ' + tag + '.csv'
	if os.path.exists(os.path.join(setpath, datasheet)):
		obj_ledger = pd.read_csv(os.path.join(setpath, datasheet))
		obj_ledger['Shape mode'] = pd.Series(idx)
		obj_ledger['Distance from cluster center'] = pd.Series(fit)
		for idxx,modegood in enumerate(kwargs['goodness'].T):
			obj_ledger['Probability of shape mode '+str(idxx+1)] = pd.Series(modegood)
		# for idxx,D in enumerate(kwargs['D'].T):
		# 	obj_ledger['Shape mode distance'+str(idxx+1)] = pd.Series(D)
		obj_ledger.to_csv(os.path.join(setpath, datasheet), index=False)
	else:
		d = {'Shape mode': pd.Series(idx), 'Distance from cluster center': pd.Series(fit)}
		obj_ledger = pd.DataFrame(data=d)
		for idxx,modegood in enumerate(kwargs['goodness'].T):
			obj_ledger['Probability of shape mode '+str(idxx+1)] = pd.Series(modegood)
		obj_ledger.to_csv(os.path.join(setpath, datasheet), index=False, columns=["Shape mode", "Distance from cluster center"])
