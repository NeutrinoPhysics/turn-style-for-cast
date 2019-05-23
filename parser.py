import os
import numpy as np
import pandas as pd
import argparse
import environment as env


def parseStation(sid):
	"""
	sid : str : station id
		the Unit label in raw file, i.e. "R324"
	"""
	sf = df[df.Unit==sid]
	sf = sf.drop(['Unit'], axis=1)
	filename = sid+'.csv'
	print('parsing', filename)
	sf.to_csv(os.path.join(env.TempFolder, filename))
	del sf
	return 



if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(description='Parse the MTA 2016 turnstyle data.')
	parser.add_argument('id', help='the station id, i.e. the Unit value in raw data file, \
						if you want a particular station. Otherwise, run all stations with "all"',\
						type=str, default='R324')
	args = parser.parse_args()

	dfile = os.path.join(env.DataFolder, env.datafiles['raw'])

	# --- import tabular data as a dataframe
	print('Reading raw data file. Storing as pandas DataFrame')
	df = pd.read_csv(	filepath_or_buffer=dfile,
					low_memory=True,
					usecols=['Unit', 'SCP', 'Date', 'Time', 'Description', 'Entries'],
					infer_datetime_format=True,
					parse_dates={'Datetime':['Date','Time']}, 
					index_col=['Datetime'],
					engine='c'
					)


	# --- drop the 'RECOVR AUD' readings
	df = df[df.Description=='REGULAR']
	df = df.drop(['Description'], axis=1)

	# --- parse all files unless user-specified test
	#df.rename(columns={'C/A': 'Id'}, inplace=True)
	#stationId = np.unique(df.Id) if args.id=='all' else np.array([args.id])
	stationId = np.unique(df.Unit) if args.id=='all' else np.array([args.id])

	# --- parse data frame per by station id
	list(map(parseStation, stationId))

