# built-in libraries
import os
# external libraries
import pandas as pd


def update_csv(idx, fit, ui, cellornuc, setpath):
	print('## recordidx.py')
	if cellornuc == 'ch1':
		ledgername = ui['ch1'][0] + '_registry.csv'
	else:
		ledgername = ui['ch2'][0] + '_registry.csv'
	if os.path.exists(os.path.join(setpath, ledgername)):
		obj_ledger = pd.read_csv(os.path.join(setpath, ledgername))
		obj_ledger['Shape mode']=pd.Series(idx)  # write
		obj_ledger['Contour fit']=pd.Series(fit)
		obj_ledger.to_csv(os.path.join(setpath, ledgername), index=False)
	else:
		d = {'Shape mode':pd.Series(idx),'Contour fit':pd.Series(fit)}
		obj_ledger = pd.DataFrame(data=d)
		obj_ledger.to_csv(os.path.join(setpath, ledgername), index=False, columns = ["Shape mode", "Contour fit"])
