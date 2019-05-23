# MTA Ridership Forecasting



## Layout

### Directories

MTA/           : the main folder
 |__ Docs/     : contains MTA turnstile data documentation and overview
 |__ Data/     : turnstile csv data file goes here
 |__ Temp/     : temporary data files for cleaning go here
 |__ Models/   : contains model pickle files and prediction dataframes
 |__ Figures/  : plots

### Files

* README.md       : this file
* environment.py  : contains global variables and filepath names
* utils.py        : contains useful functions used in several scripts
* lstm.py         : the neural network building, fitting and testing routines
* arima.py	  : the auto-regressive integrated moving average routines

* parser.py       : reads the raw data file and parses it into chunks, each corresponding to a unique station unit
* station.py      : preprocesses each individual station file
* clean.py        : cleans, reduces and interpolates missing data

* predictLSTM.py  : predict a time series with a neural network
* predictARIMA.py : predict a single, univariate time series using ARIMA
* parallel.py     : launch multiple univariate time series in parallel

* compare.py      : a short script for comparing the forecasts from the two algorithms
* pipeline.py     : the pipeline wrapper script for launching pre-processing and forecasting



## Dependencies

* Python
* NumPy
* Pandas
* Keras (Tensorflow)
* Scikit-Learn
* StatsModels
* Matplotlib
* Tmux (optional, if and when multiprocessing)


## Instructions


### Setup

1/ First, download the MTA turnstile data (csv format) and store it in the 'Data/' directory.

2/ Modify the 'raw' file name accordingly in the 'environment.py' file. The other two ('parsed' and 'clean') will get created throughout the pipeline. You can also tweek the few global parameters such as the roll-over detection threshold. 

3/ Make sure you have recent version of python (v2.7, v3.0) installed as well as all the packages listed above

/!\ Note: If you do not want to download the raw data file, you can skip the first two steps in the subsequent instructions on data cleaning. 

At this point, you may launch the entire pipeline by running in the shell prompt:

```bash
python3 pipeline.py
```

The following describe each intermediary step in the pipeline.




### Data Cleaning

1/ Parsing the raw data.

```bash
python3 parser.py all
```

This reads the raw datafile, stores the contents in a data frame and parses the contents according to the subway station units, along with each one's turnstiles subunit/channel/position. For each unit, the dataframe chunk is stored in a file containing its name (i.e. 'R324.csv' for unit R324) in the 'Temp/' directory. This allows for individual station's data inspection. Only the cumulative entries and date-time are stored. 


2/ Preprocess each station's turnstile substation.

```bash
python3 station.py
```
For each station file stored in the 'Temp/' directory (created on the previous step), the data is preprocessed in the following manner:

* organize by individual turnstiles (the 'SCP' label)
* sort by date-time to get cumulative entries
* detect any instances of roll-over digit reset, and fix accordingly
* resample (re-bin) cumulative entries by individual day
* convert cumulative entries into daily total entries for that turnstile
* sum daily entries over all turnstile units at that particular station

Once this procedure has been applied on all stations, they are concatenated into a new dataframe comprising of the daily total entries for each station, and sotred into the 'parsed' data file in the 'Data/' directory.

3/ Clean the parsed data

```bash
python3 clean.py
```

The newly created file has a lot of missing data. The following steps are taken to account for most of them:

* ditch the stations that have too much or only missing data
* remove outliers with sigma-clipping
* identify the 3 data gaps (June, July, December)
* apart from the 3 gaps, remove those that have too much consecutive days of missing data
* resample the time series into week number (1 - 52) and week day (Sun - Sat)
* fill in the missing data on a particular day by combining the expectation on that week day and on that week number

The 'clean' data file is then stored in the 'Data/' directory. 



### Fitting and Forecasting

Once the data is cleaned and stored as a dataframe, we may move on the machine learning part of the pipeline.
So far, I have implemented two algorithms for the forecasting: an ARIMA and an encoder-decoder LSTM. These are stored in the 'arima.py' and 'lstm.py' libraries, respectively. 

You may forecast with one, or both, by running the 'predictARIMA.py' and/or 'predictLSTM.py' scripts.
They take in several user-specified arguments.

They have been written to both do the fitting of the models and parameter optmization; as well as the forecasting / validation step.
The former is quite computationally expensive, and I do not recommend running it at this point, although with some minor tweeks, it could be made more efficient.
I suggest running the scripts in their validation configurations, where the hyper-parameters (for the LSTM) or orders (for the ARIMA) are fixed.



A/ ARIMA

If you do wish to run the ARIMA to model and forecast a single time series, run

```bash
python3 predictARIMA.py [series] [True/False] [p d q]
```

where [series] is the integer corresponding to the station id. For instance, the value 9 will make the code take the 10th (or 9th in python notation) column in the cleaned dataframe, which correspond to the daily entries for the unit at that address.
The default value is -1 and corrresponds to the daily entries summed over all the stations (useful for question 1). 


To tackle question 2, we would need to iterate the aforementioned script hundreds of times. To anticipate this, I wrote a script that iteratively launches batches of 'predictARIMA.py' on parallel processors. Not very useful when doing so from a personal laptop. However, if you have computational resources at hand, it might be useful to launch the 'parallel.py' script. I *strongly* recommend opening a TNUX session for doing so:

```bash
tmux new -s [my-session]
python3 parallel.py
```

/!\ NOTE:
At this point, this code is still under  construction.


B/ LSTM

If you do wish to run the LSTM encoder-decoder, 

```bash
python3 predictLSTM.py [series] [True/False]
```

where [series] is, here again, the integer corresponding to the station id, with -1 being the sum over all stations for daily entries. The second parameter to pass is a boolean that shoul be set to True if you want to fine-tune amongst many parameters. The 'environment.py' file contains the rudimentary parameter spaces on which to explore. Ideally, these parameter spaces could incorporate more values.

When only evaluating the "best" fitted model, this boolean should be False. Hence,

```bash
python3 predictLSTM.py -1 False
```

will train the network with pre-determined parameters on the full training/testing (pre-december) set, comprising of the time series of the total number of entries along all stations, and then score its forecast on the validation set.

The LSTM's hyper-parameters comprise of:

* the consecutive days to train on in the past
* the consecutive days to predict moving forward, knowing that past
* epoch, batch size, train/test split ratio
* number of units in the LSTM layer




### Inspection

The different scripts automatically generate some figures in the 'Figures/' directory for inspection. Many more features are intended to be implemented.
The 'compare.py' script serves as a rough draft for post-processing. Specifically, it compares the forecasts on the validation set (december daily usage) once both an ARIMA and neural network have made their respective predictions. It will be meant to be used in a Jupyter Notebook or an interactive python (IPython) shell; but can stillbe run as a standalone executable:


```bash
python3 compare.py
```



## Issues


Mainly GPU memory overload achieved when iteratively training neural networks from the 'predictLSTM.py' script. Issue is due to Kudas session not being released at each generation of a new model.




