"""
Meant to be a dictionary file for global vriables, hyper parameter space,
file names and paths, etc.
"""

import os
import numpy as np

# --- array of week day strings
weekdayName   = np.array(['sun', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat'])

# --- rollover counter threshod value
rollover_jump = 5e6
# 5 million is about half the population of NYC
# so it's unrealistic a spike in entries above
# that number would result from actual usage



# ======= filenames and paths
DataFolder 	= os.path.join(os.getcwd(), 'Data')
TempFolder 	= os.path.join(os.getcwd(), 'Temp')
ModelFolder = os.path.join(os.getcwd(), 'Models')
FigFolder 	= os.path.join(os.getcwd(), 'Figures')

datafiles = {
	'raw'    : 'Turnstile_Usage_Data__2016.csv',
	'parsed' : 'Daily_Entries_per_Station.csv',
	'clean'  : 'Daily_Entries_Station_Clean.csv',
	'hyper'  : 'hyper.npy'
	 		}
# =================================



# ======= Hyper Parameters for LSTM encoder-decoder
n_output    = [1, 2, 3]      # how many consecutive days to predict 
n_input     = [7, 14, 21]    # how many prior consecutive days to train on 
n_epochs    = [20, 30, 40]   # how many epochs to gradient descent
batch_size  = [15, 20, 25]   # how many samples to fit before learning
split_ratio = [20, 33, 50]   # test / validation split ratio in %
layers      = [64, 128, 256] # number of units for RNN layer

hypers = np.array([ n_output, n_input, n_epochs, batch_size, split_ratio, layers ], dtype=int)
# ====================================================





# ======= Order Parameters for ARIMA
p = [1, 2, 3] # trend autoregression order
d = [0, 1, 2] # trend difference order
q = [3, 4, 5] # trend moving average order
torder = [p,d,q]

"""
P = [0, 1, 2] # seasonal autoregression order
D = [0, 1, 2] # seasonal difference order
Q = [0, 1, 2] # seasonal moving average order

m = [7] # number of time steps for a signle seasonal period

sorder = [P,D,Q,m]
"""
# ===========================================================