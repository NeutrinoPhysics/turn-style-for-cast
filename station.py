import os
import numpy as np
import pandas as pd
import environment as env
import functools as ft


def DailyEntries(station):


	def parseSub(subid):
	
		# --- sub unit data frame
		sf = data[data.SCP==subid]
		sf = sf.drop(['SCP'], axis=1)
		sf = sf.sort_values(by='Datetime',ascending=True)
		sf.set_index(keys='Datetime', drop=True, inplace=True)
		sf = sf.resample('D').max()
		sf.rename(columns={'Entries': subid}, inplace=True)

		return sf


	filename = station+'.csv'
	dfile = os.path.join(env.TempFolder, filename)
	print('parsing station', station)

	data = pd.read_csv(	filepath_or_buffer=dfile,
					low_memory=True,
					infer_datetime_format=True,
					parse_dates=['Datetime'],
					engine='c'
					)
	subs = np.unique(data.SCP.values)
	dfs = list(map(parseSub, subs))
	units_cumul = dfs[0].join(dfs[1:]) if len(dfs)>1 else dfs[0]

	del data

	# --- identify roll over occurance (index) and location if any
	roll_over = (units_cumul.diff().abs()>env.rollover_jump).values
	if roll_over.sum() > 0:
		rollover_row, rollover_col = np.where(roll_over)
		for ro in range(rollover_col.size):
			units_cumul.iloc[rollover_row[ro]:, rollover_col[ro]]\
			+= units_cumul.iloc[rollover_row[ro]-1, rollover_col[ro]]
		print(rollover_col.size,'roll over detected')


	# --- now we're ready to differentiate
	units   = units_cumul.diff()
	entries = units.sum(axis=1, skipna=False)
	entries = pd.DataFrame({'Datetime':entries.index, station:entries.values})
	entries.set_index(keys='Datetime', drop=True, inplace=True)

	return entries



stationID = [s.split('.csv')[0] for s in os.listdir(env.TempFolder)]

entries = list(map(DailyEntries, stationID))
daily = entries[0].join(entries[1:])

# --- shift values one day earlier
daily = daily.shift(-1)

# --- drop last day since we don't have data
daily = daily.iloc[:-1,:]

daily.to_csv(os.path.join(env.DataFolder, env.datafiles['parsed']))








