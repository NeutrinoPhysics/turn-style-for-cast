"""
Launch several instances of the 'predictARIMA.py' script in parallel.
This is useful for forecasting the daily subway usage at every station.
Strongly recommended to launch using a TMUX session.
"""

from __future__ import print_function
from __future__ import division

import os
import pandas as pd
import numpy as np
import environment as env

import multiprocessing
from multiprocessing import Pipe, Process




def launch(cols, conn):
	for nc in cols:
		command = "python3 predictARIMA.py "+ str(nc) +" True 2 0 5"
		os.system(command)
	conn.send(cols[0])
	conn.close()
	return




# --- get number of columns from dataframe
dfile = os.path.join(env.DataFolder, env.datafiles['clean'])
df = pd.read_csv(filepath_or_buffer=dfile)
n_col = df.shape[1]
#n_col = 5


# --- specify number of parallel processors
n_proc = 4
batch_size = n_col//n_proc
extra = n_col%n_proc


# --- split list of columns into batches for each processor
batches = [range(k*batch_size, (k+1)*batch_size) for k in range(n_proc-1)]

# --- last processor gets the remaining columns
batches.append(range((n_proc-1)*batch_size, n_proc*batch_size+extra))

print('using {} processors.'.format(n_proc))
print('batch size {}.'.format(batch_size))


# --- multiprocessing
parents, childs, ps, lc = [], [], [], []

for ip in range(n_proc):
	ptemp, ctemp = Pipe()
	parents.append(ptemp)
	childs.append(ctemp)
	ps.append(Process(target=launch, args=(batches[ip], childs[ip])))
	ps[ip].start()

for ip in range(n_proc):
	lc.append(parents[ip].recv())
	ps[ip].join()
	print('process {0} rejoined\r'.format(ip)),
	sys.stdout.flush()

