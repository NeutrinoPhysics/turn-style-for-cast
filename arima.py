"""
The Auto-Regressive Model Builder, Fitter, and Evaluator.
A library useful mainly for the predictARIMA script
"""

import os
import numpy as np
import pandas as pd
import utils

import environment as env
import itertools
import functools as ft


from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.arima_model import ARIMA

import matplotlib as mpl
import matplotlib.pyplot as plt





def PrepareData(which=-1, normalize=False):
	"""
	Specify which time series to consider

	which 	  : int  : the column number, -1 for entries over all stations
	normalize : bool : whether to reduce series by mean and std
	"""
	dfile = os.path.join(env.DataFolder, env.datafiles['clean'])
	#dfile = os.path.join(env.DataFolder, env.datafiles['clean'].replace('_Clean', '_Clean_Gaps'))
	df = pd.read_csv(filepath_or_buffer=dfile,
			low_memory=True,
			infer_datetime_format=True,
			index_col=['Datetime'],
			engine='c'
			)

	ts = df.sum(axis=1) if which==-1 else df.iloc[:,which]
	ts[ts<1] = np.nan

	if normalize:
		ts -= ts.mean(axis=0)
		ts /= ts.std(axis=0)
		ts = np.round(ts, 3)
	else:
		ts = ts.astype(int)

	return ts




def SampleSplitter(data, split_at):
	"""
	Split the dataset into training and testing samples
	"""
	train = data[data.index<split_at]
	test  = data[data.index>=split_at]
	
	return train, test




def Delta(series, period):
	"""
	diffenrentiate time series by period
	"""
	ds = series - np.roll(series, period)
	return ds[period:]

# --- reverse-back function
Reverse = lambda past, pred, period: pred + past[-period]







def ForecastARIMA(ts, train_set, test_set, order, lag, bias=0, save=False):

	# --- intitiate past series 'ps' and predictions y_hat
	ps   = train_set.copy()
	yhat = np.zeros(test_set.size, dtype=float)

	for t in range(len(test_set)):

		# --- differentiate the series
		ds = Delta(ps, lag)

		# --- model: autoregressive integrated moving average
		model = ARIMA(ds, order=order)
		fit   = model.fit(trend='nc', disp=0)
		
		# --- only get prediction for the very next time step
		pred = fit.forecast()[0]

		# --- reverse back to un-differentiated 
		pred = Reverse(ps, pred, lag) + bias

		# --- walk-forward validation step
		yhat[t] = pred
		ps = np.hstack((ps, test_set[t]))

		if save:
			fit.save(os.path.join(env.ModelFolder, 'arima.pkl'))

	return yhat







def OptimizeParam(series, train_set, test_set, lag):
	"""
	walk through ARIMA order parameter space one-by-one
	this is only useful when trying to optimize.
	you can skip this if you know your model parameters 
	"""

	#yhat = list(map(ft.partial(Forecast, series), order_combination))
	#yhat = np.array(yhat)
	#error = utils.rmse(test, yhat)
	#return error

	# --- get order parameters from environment file
	orderspace = env.torder
	order_combination = [x for x in itertools.product(*orderspace)]

	# --- remove senseless elements. At leats p>0 or q>0 
	order_combination = [x for x in order_combination if (x[0]>0 or x[2]>0)]

	order_trend, preds = [], []

	for c in range(len(order_combination)):
		# --- must include a try statement b/c most models
		# fail to converge
		try:
			yhat = ForecastARIMA(series, train_set, test_set, order_combination[c], lag)
			order_trend.append(order_combination[c])
			preds.append(yhat)
		except:
			continue

	order_trend = np.array(order_trend)
	yhat = np.array(preds)

	return order_trend, yhat






def GetScores(actual, pred):
	"""
	get an RMS error for each model
	"""
	dim = len(pred.shape)

	if dim==1:
		# --- dont account for NaNs
		valid_yhat = ~np.isnan(pred)
		score = utils.rmse(actual[valid_yhat], pred[valid_yhat])
	
	else:

		score = []

		for r in range(pred.shape[0]):
			# --- dont account for NaNs
			valid_yhat = ~np.isnan(pred[r])
			score.append(utils.rmse(actual[valid_yhat], pred[r, valid_yhat]))

		score = np.array(score)

	return score




def CheckStationarity(series):
	"""
	Print out test statistics summary.
	Null Hypothesis (H0): series is non-stationary.
	Reject H0 if:
			* p-value < 0.01
			* test-statistic < 1% critical value
	"""

	result = adfuller(series, autolag='AIC')
	print('ADF Statistic: %f' % result[0])
	print('p-value: %f' % result[1])
	print('Critical Values:')
	for key, value in result[4].items():
		print('\t%s: %.3f' % (key, value))

	return (result[1]<0.01) * (result[0]<result[4]['1%'])




def PlotCorrelograms(series):
	"""
	plot the auto-correlation function (ACF)
	and its partial counterpart (PACF)
	to visually inspect the auto regression lag order P
	and moving average order Q
	"""
	fig, ax = plt.subplots(2,1,figsize=(8,8))
	fig = plot_acf(series.dropna(), lags=50, ax=ax[0])
	fig = plot_pacf(series.dropna(), lags=50, ax=ax[1])
	plt.xlabel('lag', fontsize=12)
	plt.savefig(os.path.join(env.FigFolder, 'correlograms.png'))
	plt.close()




