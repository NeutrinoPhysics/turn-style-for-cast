import os
import numpy as np
import pandas as pd
import utils
import arima
import environment as env
import matplotlib as mpl
import matplotlib.pyplot as plt



arima_file = os.path.join(env.ModelFolder, 'sum_pred_arima.csv')
lstm_file  = os.path.join(env.ModelFolder, 'sum_pred_lstm.csv')
dfa = pd.read_csv(filepath_or_buffer=arima_file, infer_datetime_format=True, index_col=['Datetime'],engine='c')
dfr = pd.read_csv(filepath_or_buffer=lstm_file, infer_datetime_format=True, index_col=['Datetime'],engine='c')
dfa['y_hat_lstm'] = dfr.y_hat


# --- get root mean square errors
score_arima = np.round(arima.GetScores(actual=dfa.y_obs, pred=dfa.y_hat), 3)
score_lstm = np.round(arima.GetScores(actual=dfa.y_obs, pred=dfa.y_hat_lstm), 3)



# --- plot the forecast on validation set (december)
fig = plt.figure(figsize=(12,6))
plt.plot(dfa.y_obs, linewidth=2.5, color='k', label='actual')
plt.plot(dfa.y_hat, linewidth=1.5, color='darkorange', alpha=0.5, label='ARIMA forecast, rmse = '+str(score_arima))
plt.plot(dfa.y_hat_lstm, linewidth=1.5, color='darkcyan', alpha=0.5, label='LSTM forecast, rmse = '+str(score_lstm))
plt.tick_params(axis='both', top=False, bottom=True, right=False, left=True, labelbottom=True, labeltop=False, labelleft=True, labelright=False, labelsize=12)
plt.xticks(rotation=-20)
plt.ylabel('total daily entries, normalized', fontsize=16)
plt.legend(frameon=False, loc=3, fontsize=12)
plt.savefig(os.path.join(env.FigFolder, 'forecast_compare.png'))
plt.close()



# --- plot the residuals' distribution
residuals = pd.DataFrame({'res_arima': dfa.y_hat-dfa.y_obs, 'res_lstm': dfa.y_hat_lstm-dfa.y_obs})
print(residuals.describe())


plt.figure(figsize=(8,6))
residuals.plot(kind='kde', ax=plt.gca(), color=['darkorange', 'darkcyan'])
plt.title('distribution of residuals', loc='left')
plt.xlabel('standard deviations away from observation')
plt.savefig(os.path.join(env.FigFolder, 'residual_error_distrib.png'))
plt.close()
