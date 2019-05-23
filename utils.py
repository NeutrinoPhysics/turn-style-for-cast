"""
Some useful functions for data processing 
"""

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA


# --- the root mean square error between predicted values and ground truth
rmse = lambda y_obs, y_hat: np.sqrt(mean_squared_error(y_obs, y_hat))


def reduce(data):
	"""
	normalize to get z-curve (analogous to z-score)
	"""
	red = data.copy()
	red -= red.mean(axis=0)
	red /= red.std(axis=0)
	return red


def smooth_method_1(lca, fs=10):
	"""
	Select only the 'fs' lowest modes in the FOurier transform.
	Then inverse transform to get rid of high-frequency noise
	"""
	rft = np.fft.rfft(lca)
	if len(lca.shape)==2:
		rft[:,fs:]=0.
	else:
		rft[fs:]=0.
	return np.fft.irfft(rft)


def smooth_method_2(lca, n_harm):
	"""
	Select the 'n_harm' highest coefficients in the Fourier Transform.
	Then, reconstruct signal from inverse transform, filling in the gaps
	"""

	n = lca.size
	t = np.arange(lca.size)

	p = np.polyfit(t, lca, 1)
	lca_notrend = lca - p[0] * t
	lca_freqdom = np.fft.fft(lca_notrend)
	f = np.fft.fftfreq(n) 
	indexes = list(range(n))

	indexes.sort(key = lambda i: np.absolute(f[i]))
	t = np.arange(53)
	restored = np.zeros(n)

	for i in indexes[:1 + n_harm * 2]:
		ampli = np.absolute(lca_freqdom[i]) / n
		phase = np.angle(lca_freqdom[i])
		restored += ampli * np.cos(2 * np.pi * f[i] * t + phase)

	return restored + p[0] * t




def decompose(lca, co=2):
	"""
	Decompose a sample using the first 'co' principal components.
	Returns the eigenvectors, eigenvalues, reconstructed sampes and explained variance
	"""
	model = PCA(n_components=co)
	X = model.fit_transform(lca)
	model.fit(lca)
	var = model.explained_variance_ratio_
	pc = model.components_
	pc /= pc.std(axis=1, keepdims=True)

	def pc_proj(k):
		return np.dot(lca, pc[k].T) / (np.sqrt(np.sum(lca**2, axis=1)) * np.sqrt(np.sum(pc[k]**2)))

	coef = np.asarray(list(map(pc_proj, np.arange(co))))

	# --- reconstructed light curve from principal components
	rlc = np.dot(coef.T, pc)

	return pc, coef, rlc, var
