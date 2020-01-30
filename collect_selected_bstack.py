# built-in libraries
import os
# external libraries
import pandas as pd
from tkinter import END


def collect_seleced_bstack(csv, buildmodel, entries):
	print('## collect_selected_bstack.py')
	if buildmodel:
		ui = pd.read_csv(csv)
		setpaths = ui['set location']
		ch1 = ui['ch1']
		ch2 = ui['ch2']
		c1_stacks = []
		c2_stacks = []
		for setidx, setpath in enumerate(setpaths):
			pickles = [_ for _ in os.listdir(setpath) if _.lower().endswith('pickle')]
			c1_stack = [pd.read_pickle(os.path.join(setpath, pkl)) for pkl in pickles if ch1[setidx] in pkl]
			c2_stack = [pd.read_pickle(os.path.join(setpath, pkl)) for pkl in pickles if ch2[setidx] in pkl]
			c1_stacks = c1_stacks + c1_stack
			c2_stacks = c2_stacks + c2_stack
		try:
			df_c1 = pd.concat(c1_stacks, ignore_index=True)
		except:
			print('CSV file is empty')
			df_c1 = None
			entries['Status'].delete(0, END)
			entries['Status'].insert(0, 'CSV file is empty')
		df_c2 = pd.concat(c2_stacks, ignore_index=True)
	else:
		raise NameError('collect_seleced_bstack is only for buildmodel')
		return

	return df_c1, df_c2