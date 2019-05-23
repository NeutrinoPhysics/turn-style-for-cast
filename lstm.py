"""
The Neural Network Builder, Fitter, and Evaluator.
A library useful mainly for the predictLSTM script
"""

import os
import environment as env
import numpy as np
import utils
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Flatten, RepeatVector, TimeDistributed






def GradeNetwork(dataset, hypers, save=False):
	# ---parse the hyperparameters
	n_output, n_input, \
	n_epochs, batch_size, \
	split_ratio, layers = hypers

	split_ratio/=100
	split_ratio = np.round(split_ratio, 2)

	layers = [layers, int(layers/2)]

	train, test = SampleSplitter(data=dataset, split_at='2016-12-01', n_out=n_output)

	forecast, score, scores = LaunchRNN(
		train = train,
		test  = test,
		n_in  = n_input,
		n_out = n_output,
		epo   = n_epochs,
		bs    = batch_size,
		sr    = split_ratio,
		lyrs  = layers,
		save  = save
		)

	return forecast, score, scores






def LaunchRNN(train, test, n_in, n_out, epo, bs, sr, lyrs, save):

	# --- thread and fit model
	#tf.keras.backend.set_session(sess)
	rnn_rnn = ModelFitter(train, n_in, n_out, epo, bs, sr, lyrs)

	# --- ready past consecutive days
	past = [x for x in train]
	y_hat = []
	
	# --- walk-forward loop
	#     add ground truth to past list to predit next day
	for i in range(len(test)):
		y_hat.append(Predict(model=rnn_rnn, past=past, n_in=n_in))
		past.append(test[i, :])
	
	# --- grade
	y_hat = np.array(y_hat)
	score, scores = Error(truth=test[:, :, 0], pred=y_hat)
	
	# --- clear the current model
	if save:
		rnn_rnn.save(os.path.join(env.ModelFolder, 'lstm.h5'))

	del rnn_rnn
	tf.keras.backend.clear_session()

	return y_hat, score, scores









def SampleSplitter(data, split_at, n_out):
	"""
	Split the dataset into training/validation and testing samples
	NOTE: for the chosen architecture, the input vectors should be of the shape:
	[N/n_output, n_output, 1] and [M/n_output, n_output, 1] respectively
	where N and M are the pre-december and post-december sample sizes 
	"""

	xi    = data.index
	yi    = data.values

	si = np.where(xi==split_at)[0][0]
	train, test = yi[:si], yi[si:]

	cut_train = train.shape[0]%n_out
	cut_test  = test.shape[0]%n_out

	train = train[cut_train:]
	if cut_test>0:
		test  = test[:-cut_test]

	# --- reshape to have correct dims for RNN
	train = train.reshape(int(train.shape[0]/n_out), n_out, 1)
	test = test.reshape(int(test.shape[0]/n_out), n_out, 1)
	
	return train, test






def Error(truth, pred):

	scores = []

	for i in range(truth.shape[1]):
		err = utils.rmse(y_obs=truth[:, i], y_hat=pred[:, i])
		scores.append(err)
	
	s = 0
	for row in range(truth.shape[0]):
		for col in range(truth.shape[1]):
			s += (truth[row, col] - pred[row, col])**2

	score = np.sqrt(s / (truth.shape[0] * truth.shape[1]))
	score = np.round(score[0], 6)

	return score, scores






def NextSample(vec_train, n_in, n_out):

	# --- adequately shape
	shape = vec_train.shape

	# --- shape into [samples * timesteps, features] 
	data = vec_train.reshape((shape[0]*shape[1],shape[2]))

	# --- initialize features and labels
	features, labels = [], []

	# (overlapping) batch number
	bn = 0
	for _ in range(len(data)):
		# --- define the end of the input sequence
		fea_upper = bn + n_in
		lbl_upper = fea_upper + n_out

		if lbl_upper < len(data):
			# --- each feature batch is 28 x 1 vector 
			batch = data[bn:fea_upper, 0]
			batch = batch.reshape((len(batch), 1))
			# --- each label is 1 x 1 vector
			features.append(batch)
			labels.append(data[fea_upper:lbl_upper, 0])
		bn += 1

	# --- collect all overlapping batches.
	#     feature vector is shape (334 - 28 - 1) x 28 x 1
	#     label vector is shape (334 - 28 - 1) x 1
	return np.array(features), np.array(labels)



 
def ModelFitter(train, n_in, n_out, epo, bs, sr, layers):

	tf.keras.backend.clear_session()

	training_features, training_labels = NextSample(train, n_in, n_out)
	y_shape = training_labels.shape
	x_shape = training_features.shape

	Ndate, Nfeat, Nout = x_shape[1], x_shape[2], y_shape[1]

	training_labels = training_labels.reshape((y_shape[0], y_shape[1], 1))

	model = keras.Sequential([
		LSTM(layers[0], activation='relu', input_shape=(Ndate, Nfeat)),
		RepeatVector(Nout),
		LSTM(layers[0], activation='relu', return_sequences=True),
		TimeDistributed(Dense(layers[1], activation='relu')),
		TimeDistributed(Dense(1))
			])
	
	model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
	model.fit(training_features, training_labels, epochs=epo, batch_size=bs, validation_split=sr)

	return model



def Predict(model, past, n_in):

	new_past  = np.array(past)
	shape 	  = new_past.shape
	new_past  = new_past.reshape((np.prod(shape[:2]), shape[-1]))

	input_x   = new_past[-n_in:,0]
	input_x   = input_x.reshape((1, len(input_x), 1))
	
	# --- make prediction for next instance in time series
	yhat = model.predict(input_x)

	return yhat[0]
 



