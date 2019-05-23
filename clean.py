import os
import numpy as np
import pandas as pd
import functools as ft
import utils
import environment as env

import matplotlib as mpl
import matplotlib.pyplot as plt
#plt.ion()


def SigmaClip(df, ncol):
	"""
	remove outliers by a sigma clipping method
	returns array with outliers masked out
	
	df: dataframe
	ncol: column number (station) to apply the routine to
	"""
	sig = 3
	arr  = df.iloc[:,ncol].values 

	norm = np.ma.masked_invalid(arr)
	med  = np.ma.median(norm)
	std  = norm.std()

	norm.mask = np.abs(norm-med) > sig*std
	norm.data[norm.mask]=np.nan

	return norm.data



def count_nan(df, ncol):
	"""
	counts the amount of NaNs in column ncols of dataframe df
	"""
	return len(df.iloc[:,ncol]) - df.iloc[:,ncol].count()



def prune_empty(max_nans):
	"""
	returns a list of colums that don't have a NaN count exceeding max_nans 
	"""
	nan_in_columns   = list(map(ft.partial(count_nan, de), np.arange(de.shape[-1])))
	return np.where(np.array(nan_in_columns)<max_nans)[0]




def common_missing(df):
	"""
	returns index boolean on location of missing data
	common to all series, i.e. the 3 large gaps
	"""
	val    = df.values
	valnan = np.isnan(val)
	common = np.product(valnan, axis=1)
	return common.astype(bool)





def prune_consecutive_missing(con_len, ncol):
	"""
	checks if column 'ncol' has consecutive nans
	that exceed 'con_len' 
	"""
	col = dm.iloc[:,ncol]
	nan_con_len = col.isnull().astype(int).groupby(col.notnull().astype(int).cumsum()).sum().values
	nan_len 	= np.unique(nan_con_len)
	nan_len_max = nan_len.max()
	return True if nan_len_max < con_len else False





# --- load parsed file
dfile = os.path.join(env.DataFolder, env.datafiles['parsed'])
de = pd.read_csv(filepath_or_buffer=dfile,
				low_memory=True,
				infer_datetime_format=True,
				index_col=['Datetime'],
				engine='c'
				)



# --- get rid of clear outliers
de[(np.abs(de)>=1e5)] = np.nan

# --- select only time series that dont have too much nans
cols_to_keep = prune_empty(max_nans=50)
de = de.iloc[:,cols_to_keep]

# --- find common indices of missing dates and drop them
cm = common_missing(df=de)
dm = de.loc[~cm]

# --- prune off columns missing more than 7 consecutive days of data,
#     not counting the 3 large gaps 
keep = list(map(ft.partial(prune_consecutive_missing, 7), np.arange(dm.shape[-1])))
depd = de.iloc[:,keep]

# --- get rid of negative values
depd[(depd<0)] = np.nan


# --- apply one iteration of 3-sigma clipping
clip = np.array(list(map(ft.partial(SigmaClip, depd), np.arange(depd.shape[-1])))).T
dep = pd.DataFrame(data=clip, index=depd.index, columns=depd.columns)



#fig = plt.figure(figsize=(10,10))
plt.imshow(dep, cmap=mpl.cm.coolwarm, interpolation=None)
plt.tick_params(axis='both', top=False, bottom=True, right=False, left=True, labelbottom=True, labeltop=False, labelleft=True, labelright=False, labelsize=12)
plt.ylabel('days since Jan. 2, 2016', fontsize=16)
plt.xlabel('station unit index', fontsize=16)
plt.title('daily entries', fontsize=16, loc='left')
plt.colorbar()
plt.savefig(os.path.join(env.FigFolder, 'daily_entries_per_station.png'))
plt.close()





# --- identify weekly trends in data

dep.index = pd.to_datetime(dep.index)
std = dep.std(axis=0)
med = dep.mean(axis=0)



# resampling per week day and week number
dep['weekday'] = pd.to_datetime(dep.index)
dep['weekday'] = dep.weekday.dt.dayofweek

dep['weeknum'] = pd.to_datetime(dep.index)
dep['weeknum'] = dep.weeknum.dt.weekofyear

wk = dep.resample('W').mean()
wd = dep.groupby(dep.weekday).mean()



# save the mean and stdv for reconstruction
std_wk = wk.std(axis=0)
med_wk = wk.mean(axis=0)
std_wd = wd.std(axis=0)
med_wd = wd.mean(axis=0)





# --- low-frequency trends:
#     smooth out the normalized station-average
#     by one of 2 Fourier transform methods 
rwk = utils.reduce(data=wk)
weeknum_trend = rwk.mean(axis=1)

# method 1: by clearing high-frequency modes
wnts_1 = utils.smooth_method_1(lca=weeknum_trend, fs=4)

# method 2: by taking the strongest mode and then reconstructing back
wnts_2 = utils.smooth_method_2(lca=weeknum_trend, n_harm=4)




fig = plt.figure(figsize=(12,6))
for r in range(rwk.shape[1]):
	plt.scatter(x=np.arange(weeknum_trend.size), y=rwk.values[:,r], marker='o', color='k', s=30, alpha=0.05)
plt.scatter(x=np.arange(weeknum_trend.size), y=rwk.values[:,0], marker='o', color='k', s=30, alpha=0.5, label='individual station')
plt.scatter(x=np.arange(weeknum_trend.size), y=weeknum_trend, marker='s', color='darkorange', s=60, alpha=0.75, label='station-averaged')
#plt.plot(wnts_1, color='darkorange', alpha=0.8, linewidth=2.5)
plt.plot(wnts_2, color='darkcyan', alpha=0.8, linewidth=2.5, label='Fourier-smoothed trend')
plt.tick_params(axis='both', top=False, bottom=True, right=False, left=True, labelbottom=True, labeltop=False, labelleft=True, labelright=False, labelsize=12)
plt.xlabel('week number', fontsize=16)
plt.ylabel('weekly entries, normalized', fontsize=16)
plt.legend(frameon=False, fontsize=12, loc=0)
plt.savefig(os.path.join(env.FigFolder, 'seasonal_trend_Fourier.png'))
plt.close()





# --- sub-week trends:
#     due to low bin count (7) and (reatively)
#     high number of samples, a simple average will do
#     if the standard deviation is low
rwd = utils.reduce(data=wd)
weekday_trend = rwd.mean(axis=1)






# --- for each missing piece of data, identify its coordinates
#     in terms of week number and week day.
day_id, sta_id = np.where(np.isnan(dep.values))
day_of_week = dep.weekday[day_id].values
week_of_year = dep.weeknum[day_id].values.astype(int)-1

trend_week = wnts_2[week_of_year] # choosing the 2nd Fourier method
trend_wday = weekday_trend[day_of_week]

# --- pad the missing data with an additive model
#     that corresponds to the smoothed interpolated week-number trend
#     plus the day number trend. 
correction = (trend_week + trend_wday).values
correction *= std.values[sta_id]
correction += med.values[sta_id]

# --- drop the last 2 columns we artificially created
red = dep.drop(['weekday', 'weeknum'], axis=1)


# --- now fill in the missing data with a function call
correction = list(correction)
idx = [(day_id[c], sta_id[c]) for c in range(day_id.size)]

def fill_missing(loc):
	x, y = idx[loc]
	red.iloc[x,y] = correction[loc]
	return

mute = list(map(fill_missing, np.arange(len(correction))))

# --- save cleaned data
red.to_csv(os.path.join(env.DataFolder, env.datafiles['clean']))





total_entries = red.sum(axis=1)*1e-6
fig = plt.figure(figsize=(12,6))
plt.plot(total_entries, color='k', alpha=0.2, linewidth=0.75)
plt.scatter(x=total_entries.index[cm], y=total_entries[cm], color='darkorange', alpha=0.8, s=20, label='reconstructed')
plt.scatter(x=total_entries.index[~cm], y=total_entries[~cm], color='darkcyan', alpha=0.8, s=20, label='real')
plt.tick_params(axis='both', top=False, bottom=True, right=False, left=True, labelbottom=True, labeltop=False, labelleft=True, labelright=False, labelsize=12)
plt.ylabel('total daily entries [millions]', fontsize=16)
plt.legend(frameon=False, loc=0, fontsize=12)
plt.savefig(os.path.join(env.FigFolder, 'totals.png'))
plt.close()





red = utils.reduce(data=red)
plt.imshow(red, clim=[-2,2], cmap=mpl.cm.coolwarm, interpolation=None)
plt.tick_params(axis='both', top=False, bottom=True, right=False, left=True, labelbottom=True, labeltop=False, labelleft=True, labelright=False, labelsize=12)
plt.ylabel('days since Jan. 2, 2016', fontsize=16)
plt.xlabel('station unit index', fontsize=16)
plt.title('daily entries, normalized by station', fontsize=16, loc='left')
plt.colorbar()
plt.savefig(os.path.join(env.FigFolder, 'daily_entries_per_station_cleaned_norm.png'))
plt.close()




