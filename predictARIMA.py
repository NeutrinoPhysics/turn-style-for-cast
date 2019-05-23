import os
import numpy as np
import pandas as pd
import utils
import arima
import argparse

import environment as env

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

import matplotlib as mpl
import matplotlib.pyplot as plt

	
argp = argparse.ArgumentParser(description='run ARIMA with feed-forward validation')
argp.add_argument('ts', help='which time series (column number). -1 for all stations combined', type=int, default=-1)
argp.add_argument('norm', help='normalize your time series by its mean and std deviation?', type=bool, default=False)
argp.add_argument('model', help='values of (p, d, q) or (-1,-1,-1) if you need to determine it', nargs=3, type=int, default=(2,0,5), metavar=('p', 'd', 'q'))
args = argp.parse_args()


# --- fetch our time series
ts = arima.PrepareData(which=args.ts, normalize=args.norm)

# --- perform Augmented Dickey Fuller statistical test
# on native time series


arima.CheckStationarity(series=ts)

# the time series itself isn't stationary.
# but if we substract its rolling average, it is !


tra = ts-ts.rolling(window=28).mean()
arima.CheckStationarity(series=tra.dropna())


# an even better stationarity is obtained when substracting
# the trend line by using seasonal decomposition

decomp = seasonal_decompose(x=ts, freq=7)
trend    = decomp.trend      # captures yearly-periodic cycles 
season   = decomp.seasonal   # captures the weekly cycle
residual = decomp.resid      # ts = res + season + trend
arima.CheckStationarity(series=residual.dropna())


# problem is: the rolling standard deviation 
# of the residuals is not constant.
# Plus, the seasonal and trend components are auto-correlated
# as is visible in when plotting their ACF

# i.e., 
#arima.PlotCorrelograms(series=trend)

# let's settle with the series time-lgged by 7 days


# --- check the correlograms to guess p and q

seasonal_period = 7
tra = arima.Delta(ts, seasonal_period)
arima.PlotCorrelograms(series=tra)


# by the looks of it, Q is somewhere around 4, 5, or 6
# and P is around 1 and 2.
# At this point, we can check which of these models best fit the series
# by setting some deviations around our best guess in 'environment.py'


# --- splitting the parent sample into initial train / test sets
training, validation = arima.SampleSplitter(ts, split_at='2016-12-01')
train, test = arima.SampleSplitter(training, split_at='2016-06-01')



if args.model == (-1, -1, -1): 

	# *** /!\ ***
	# this next step is long and computationally expansive
	# skip this if you already know your model
	orders, yhat = arima.OptimizeParam(series=ts, train_set=train, test_set=test, lag=seasonal_period)

	# --- remove models that predict too much nans
	# let's tolerate 2 max
	valid  = np.sum(np.isnan(yhat), axis=1) < 3
	orders = orders[valid]
	yhat   = yhat[valid]

	scores = arima.GetScores(actual=test, pred=yhat)


	# --- get best ARIMA model
	bg = scores.argmin()
	best_order = orders[bg]
	best_score = scores[bg]
	pp, dd, qq = best_order
	print('ARIMA ( p=',str(pp),', d=',str(dd),', q=',str(qq),' )')


	# --- summarize residuals statistics
	residuals = pd.DataFrame({'residuals': yhat[bg]-test})
	print(residuals.describe())


	# --- show the forecasted vs labeled test values
	fig = plt.figure(figsize=(12,6))
	for r in range(yhat.shape[0]):
		plt.plot(yhat[r], color='darkcyan', alpha=0.1, linewidth=1.5)
	plt.plot(test, color='darkorange', alpha=0.8, linewidth=2.5, label='actual')
	plt.plot(yhat[bg], color='darkcyan', alpha=0.8, linewidth=2.5, label='best model')
	plt.plot(yhat[bg], color='darkcyan', alpha=0.1, linewidth=1.5, label='ARIMA models')
	plt.tick_params(axis='both', top=False, bottom=True, right=False, left=True, labelbottom=True, labeltop=False, labelleft=True, labelright=False, labelsize=12)
	plt.ylabel('total daily entries', fontsize=16)
	plt.xlabel('days since Dec.1, 2016', fontsize=16)
	plt.title('RMSE:'+str(int(np.round(best_score,0)))+' ; ('+str(np.round(100*best_score/test.mean(),2))+'% of average)', loc='left', fontsize=12)
	plt.legend(frameon=False, loc=0, fontsize=12)
	plt.savefig(os.path.join(env.FigFolder, 'ARIMA'+'_'+str(pp)+'_'+str(dd)+'_'+str(qq)+'.png'))
	plt.close()



else:

	# --- this time, train on entire training set, and test on validation set
	y_hat = arima.ForecastARIMA(ts=ts,
								train_set=training,
								test_set=validation,
								order=args.model,
								#order=(2,0,5),
								lag=seasonal_period,
								bias=0,
								save=True
								)

	# --- score model on forecast vs validation set
	score = arima.GetScores(actual=validation, pred=y_hat)

	# --- write prediction vs obervation table 
	if args.ts==-1:
		ts.name = 'sum'
	filename = ts.name+'_pred_arima.csv'
	out = pd.DataFrame({'y_obs': validation, 'y_hat': np.round(y_hat,3)})
	out.index = pd.to_datetime(ts.index[-validation.size:])
	out.to_csv(os.path.join(env.ModelFolder, filename))


