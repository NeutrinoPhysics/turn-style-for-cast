import os
import numpy as np
import pandas as pd
import utils
import arima
import lstm
import argparse
import environment as env



def SetupHypers(i, j):
	"""
	Returns a specific combination of hyper-parameters for a given (i,j).
	Fixes all other hyper-parameters to their central value
	
	for example, the i=2 and j=0 combination from the parameter
	space [A = [a1, a2, a3], B=[b1, b2, b3], C=[c1, c2, c3]] will be:
	[ A=a2, B=b2, C=c1 ]
	"""
	mat = np.tile(np.array([0,1,0], dtype=bool), hyperspace.shape[0]).reshape(hyperspace.shape)
	mat[i] = np.zeros(3, dtype=bool)
	mat[i,j] = True
	hypers = hyperspace[mat]
	return tuple(hypers)


def WalkParam(data):
	"""
	walk through hyper parameter space one-by-one
	*** /!\ *** lengthy and inefficient 
	"""

	# --- initiate score array
	scores = np.zeros(hyperspace.shape)

	for i in range(scores.shape[0]): # not my proudest nest loop
		for j in range(scores.shape[1]):

			# --- get hyper parameter at i,j indices and fix all others
			hypers = SetupHypers(i,j)

			# --- score the network for that set of hyperparameters
			scores[i,j] = lstm.GradeNetwork(dataset=data, hypers=hypers, save=False)[1]

	return scores


def FineTuned(scores):
	"""
	set parameters to their "best" values
	"""
	best = scores.argmin(axis=1)
	tuned = []
	for r in zip(range(hyperspace.shape[0]), best):
		tuned.append(hyperspace[r])
	return tuned




if __name__ == '__main__':


	argp = argparse.ArgumentParser(description='run ARIMA with feed-forward validation')
	argp.add_argument('ts', help='which time series (column number). -1 for all stations combined', type=int, default=-1)
	argp.add_argument('tuning', help='True for tuning hyper-parameters, False for launching specific model', type=bool, default=False)
	args = argp.parse_args()


	if args.tuning=='True':
		
		hyperspace 	 = env.hypers

		ts = arima.PrepareData(which=args.ts, normalize=True)
		scores 		 = WalkParam(data=ts)
		tuned_params = FineTuned(scores = scores) 
		print('tuned params: ', tuned_params)

		# --- save the "fine-tuned" hyper-parameters
		np.save(os.path.join(env.ModelFolder, 'hyper.npy'), np.array(tuned_params))

	else:


		# --- find nomenclature in 'environment.py'
		tuned_hyperparameters = [1, 21, 50, 20, 20, 256]
		# alternatively:
		# tuned_hyperparameters = np.load(os.path.join(env.ModelFolder, 'hyper.npy'))


		# --- fetch our time series
		ts = arima.PrepareData(which=args.ts, normalize=True)
		training, validation = lstm.SampleSplitter(data=ts, split_at='2016-12-01', n_out=tuned_hyperparameters[0])

		# --- score the network for that set of hyperparameters
		y_hat = lstm.GradeNetwork(dataset=ts, hypers=tuned_hyperparameters, save=True)[0]
		score = lstm.Error(truth=validation, pred=y_hat)[0]

		# --- write prediction vs obervation table 
		if args.ts==-1:
			ts.name = 'sum'
		filename = ts.name+'_pred_lstm.csv'
		out = pd.DataFrame({'y_obs': validation[:,0,0], 'y_hat': np.round(y_hat[:,0,0],3)})
		out.index = pd.to_datetime(ts.index[-validation.size:])
		out.to_csv(os.path.join(env.ModelFolder, filename))



