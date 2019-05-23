"""
The full pipeline wrapper arounf the preporcessing and forcasting stages
Comment / uncomment accordingly
"""

import os


# ===== PREPROCESS

# --- read raw file and store station chunks
os.system("python3 parser.py all")

# --- read all station chunks, group by subunits, acount for rollover 
os.system("python3 station.py")

# --- clip outliers and interpolate missing values
os.system("python3 clean.py")

# =============================





# ============= TRAIN AND FORECAST

# A /  arima

# --- total entries, not normalized, fixed orders (p,d,q)
os.system("python3 predictARIMA.py -1 False 2 0 5")

# --- specific station entries, normalized, fixed orders (p,d,q)
#os.system("python3 predictARIMA.py 78 True 2 0 5")

# --- total entries, normalized, find model
#os.system("python3 predictARIMA.py -1 True -1 -1 -1")

# --- specific station entries, not normalized, find model
#os.system("python3 predictARIMA.py 46 False -1 -1 -1")




# B /   LSTM univariate encoder-decoder

# note: the entries are always normalized for the neural network

# --- total entries, trust the model
os.system("python3 predictLSTM.py -1 False")

# --- specific station entries, trust the model
#os.system("python3 predictLSTM.py 177 False")

# --- total entries, find the best fitting model (!! expensive)
#os.system("python3 predictLSTM.py -1 True")



